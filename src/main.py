import json

import numpy as np


import mutation_generator
import test_case_splitter
import os
import mutation_executor
import metallaxis
import muse
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ML_rank.Mutant_Prior import MutantPrior
from comparator import comparator_factory
from Utils.utils import write_to_file, convert_keys_to_int, append_dict_to_csv, append_scores_to_csv, getIsClass
from feature_extraction.Util.cal_stats import summary, summary_Single

if __name__ == '__main__':
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        raise ValueError('This program expects 6 or 7 command-line arguments')
    model_file = sys.argv[1]
    mutant_ratio = float(sys.argv[2])
    X_test = np.load(sys.argv[3])
    y_test = np.load(sys.argv[4])
    model_kind = sys.argv[5]
    delta = 1e-3
    last_index = max(model_file.rfind('/'), model_file.rfind('\\'))
    base_dir=model_file[:last_index]
    model_name=int(os.path.basename(os.path.normpath(base_dir)))
    if len(sys.argv) == 7:
        delta = float(sys.argv[6])
    fraction=1
    Isclass=getIsClass(model_kind)
    comparator = comparator_factory(model_kind, delta)
    print("State 1 finish!")
    tmp_dict = os.path.join(base_dir, "result_dir", "mut_dict.json")
    with open('%d.txt' % int(fraction * 100), 'w') as out_file:
        if not os.path.isfile('./workdir.tar.gz') or not os.path.isfile(tmp_dict):
            if os.path.exists('./workdir.tar.gz'):
                os.remove('./workdir.tar.gz')
            start = time.time()
            print(model_file)
            mg = mutation_generator.MutationGenerator(model_file)
            mg.apply_dup_layer()
            mg.apply_math_weight()
            mg.apply_math_bias()
            mg.apply_math_weight_conv()
            mg.apply_math_bias_conv()
            mg.apply_math_filters()
            mg.apply_math_kernel_sz()
            mg.apply_math_strides()
            mg.apply_math_pool_sz()
            mg.apply_padding_replacement()
            mg.apply_activation_function_replacement()
            # mg.apply_del_layer()
            # mg.apply_dup_layer()
            mg.apply_math_lstm_input_weight()
            mg.apply_math_lstm_forget_weight()
            mg.apply_math_lstm_cell_weight()
            mg.apply_math_lstm_output_weight()
            mg.apply_math_lstm_input_bias()
            mg.apply_math_lstm_forget_bias()
            mg.apply_math_lstm_cell_bias()
            mg.apply_math_lstm_output_bias()
            mg.apply_recurrent_activation_function_replacement()
            mg.store_dict()
            mg.close()
            end = time.time()
            print('Mutation generation took %s seconds' % (end - start))
            out_file.write('Mutation generation took %s seconds\n' % (end - start))
        else:
            print("Hello World!")
        with open(tmp_dict, 'r') as file:
            mut_dict = json.load(file)
        mut_dict = convert_keys_to_int(mut_dict)
        start = time.time()
        s = test_case_splitter.TestCaseSplitter(model_file, X_test, y_test, comparator)
        s.split()
        passDict={
            'pass':len(s.get_passing_test_outputs()),
            'fail':len(s.get_failing_test_actual_outputs())
        }
        end = time.time()
        test_split_time=end-start
        print('Test case splitting took %s seconds' % (end - start))
        out_file.write('Test case splitting took %s seconds\n' % (end - start))
        write_to_file(base_dir, mut_dict, passDict, s.loss_func)
        print('Selected %2f%% of the mutants' % (mutant_ratio * 100))
        summary_Single(model_dir=base_dir,isclass=Isclass,model_type=s.getModelType())
        mp_xgb=MutantPrior(select_ratio=mutant_ratio,file_path=os.path.join(base_dir,"all_summary.csv"))
        mp_xgb.process()
        mt_set=mp_xgb.getMutantSet()
        start = time.time()
        mt = mutation_executor.MutationExecutor(s, comparator)
        mt.set_mt_set(mt_set)
        mt.set_mutant_selection_fraction(fraction)
        mtr = mt.test()
        end = time.time()
        print('Mutation execution took %s seconds' % (end - start))
        out_file.write('Mutation execution took %s seconds\n' % (end - start))
        print('DeepMPrior selected %d mutants of %d mutants, among them %d turned out to be non-viable'
              % (mt.get_select_mutants_count(),mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))
        out_file.write('DeepMPrior selected %d mutants of %d mutants, among them %d turned out to be non-viable\n'
              % (mt.get_select_mutants_count(),mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))

        me = metallaxis.Metallaxis(mtr, mt.get_failing_tests_total_count())
        # print(mtr)
        # me.set_mutant_sus_score(mut_dict)
        me.calculate_type1_scores()
        # mut_dict = me.get_mutant_sus_score()
        print('Metallaxis - Type 1:')
        print('\tSBI Avg: %s' % me.get_avg_sbi_scores())
        print('\tSBI Max: %s' % me.get_max_sbi_scores())
        print('')
        print('\tOchiai Avg: %s' % me.get_avg_ochiai_scores())
        print('\tOchiai Max: %s' % me.get_max_ochiai_scores())
        out_file.write('Metallaxis - Type 1:\n')
        out_file.write('\tSBI Avg: %s\n' % me.get_avg_sbi_scores())
        out_file.write('\tSBI Max: %s\n' % me.get_max_sbi_scores())
        out_file.write('\n')
        out_file.write('\tOchiai Avg: %s\n' % me.get_avg_ochiai_scores())
        out_file.write('\tOchiai Max: %s\n' % me.get_max_ochiai_scores())
        print('--------------------')
        # out_file.write('--------------------\n')
        # append_scores_to_csv(me.get_max_ochiai_scores(), model_name)
        me.calculate_type2_scores()
        print('Metallaxis - Type 2:')
        print('\tSBI Avg: %s' % me.get_avg_sbi_scores())
        print('\tSBI Max: %s' % me.get_max_sbi_scores())
        print('')
        print('\tOchiai Avg: %s' % me.get_avg_ochiai_scores())
        print('\tOchiai Max: %s' % me.get_max_ochiai_scores())
        out_file.write('Metallaxis - Type 2:\n')
        out_file.write('\tSBI Avg: %s\n' % me.get_avg_sbi_scores())
        out_file.write('\tSBI Max: %s\n' % me.get_max_sbi_scores())
        out_file.write('\n')
        out_file.write('\tOchiai Avg: %s\n' % me.get_avg_ochiai_scores())
        out_file.write('\tOchiai Max: %s\n' % me.get_max_ochiai_scores())
        print('')
        print('====================')
        print('')
        out_file.write('\n')
        out_file.write('====================\n')
        out_file.write('\n')
        mu = muse.MUSE(mtr, mt.get_passing_tests_total_count(), mt.get_failing_tests_total_count())
        mu.calculate_scores()
        # print(mutant_score_dict)
        print('MUSE:')
        print('\t%s' % mu.get_scores())
        out_file.write('MUSE:\n')
        out_file.write('\t%s\n' % mu.get_scores())
        # write_to_file(base_dir,mut_dict,passDict,s.loss_func)
