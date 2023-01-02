
import os


# source data folder
# test_folder = '/home/yuhai/workspace/qa/data/race/test'
# adv_test_para_folder = '/home/yuhai/workspace/qa/data/para-race/test'
# adv_test_charswap_folder = '/home/yuhai/workspace/qa/data/adv-race/charSwap'
# adv_test_addsent_folder = '/home/yuhai/workspace/qa/data/adv-race/AddSent'


def find_results(folder, correct=True):
    # find cases which are predicted correct if correct=True
    file = os.path.join(folder, 'eval_predictions.json')
    results = {}    # id -> (pre, label)
    with open(file, 'r') as file:
        for line in file.readlines():
            id_, pre, label = line.strip().split('\t')
            if (correct and pre == label) or (not correct and (pre != label)):
                results[id_] = (pre, label)
    return results


if __name__ == '__main__':
    # race and adv-race
    race_test_root_dir = '/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/adv-race/test/'

    orig_name = 'orig/all'
    adv_names = ['charswap/all']
    adv_names = ['para-ques/all']
    adv_names = ['addsent/all']
    adv_names = ['charswap/all', 'para-ques/all']
    adv_names = ['charswap/all', 'addsent/all']
    adv_names = ['para-ques/all', 'addsent/all']
    adv_names = ['charswap/all', 'para-ques/all', 'addsent/all']


    model_name = 'bert-base-uncased'
    res_folders = {
        'base': 'stable_cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-24_23:58:13',
        'scqa': 'stable_cl_bert-epochs_4-lr_3e-5-per_bs_3-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-12_10:22:22',
    }

    # model_name = 'roberta-base'
    # res_folders = {
    #     'base': 'stable_cl_roberta-lr_3e-6-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-10_00:11:46',
    #     'scqa': 'stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.005-2022-05-10_01:34:39',
    # }

    model_name = 'albert-base-v2'
    res_folders = {
        'base': 'stable_cl_albert-epochs_10-lr_1e-6-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_19:28:26',
        'scqa': 'stable_cl_albert-epochs_8-lr_1e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-18_13:34:02'
    }
    file_name = '-'.join([name.replace('/', '-') for name in adv_names])
    good_cases_file = os.path.join(race_test_root_dir, orig_name, model_name, 'good_cases-%s.txt' % file_name)

    orig_base_folder = os.path.join(race_test_root_dir, orig_name, model_name, res_folders['base'])
    orig_scqa_folder = os.path.join(race_test_root_dir, orig_name, model_name, res_folders['scqa'])

    orig_base_correct = find_results(orig_base_folder, correct=True)
    orig_scqa_correct = find_results(orig_scqa_folder, correct=True)

    adv_base_wrongs, adv_scqa_corrects = [], []     # list of dict
    for adv_type in adv_names:
        base_folder = os.path.join(race_test_root_dir, adv_type, model_name, res_folders['base'])
        scqa_folder = os.path.join(race_test_root_dir, adv_type, model_name, res_folders['scqa'])

        base_wrong = find_results(base_folder, correct=False)
        scqa_correct = find_results(scqa_folder, correct=True)

        adv_base_wrongs.append(base_wrong)
        adv_scqa_corrects.append(scqa_correct)

    # find cases satisfied with following results
    #       orig    adv1    adv2    adv3
    # base  1       0       0       0
    # scqa  1       1       1       1

    good_case_ids = []
    for id_ in orig_base_correct:
        if id_ not in orig_scqa_correct:
            continue

        not_in_adv_base_wrong = False
        for adv_base_wrong in adv_base_wrongs:
            if id_ not in adv_base_wrong:
                not_in_adv_base_wrong = True
                break
        if not_in_adv_base_wrong:
            continue

        not_in_adv_scqa_correct = False
        for adv_scqa_correct in adv_scqa_corrects:
            if id_ not in adv_scqa_correct:
                not_in_adv_scqa_correct = True
                break
        if not_in_adv_scqa_correct:
            continue

        good_case_ids.append(id_)

    with open(good_cases_file, 'w') as f:
        f.write('0-0\torig-all-base: %s\n' % orig_base_folder + '/eval_predictions.json')
        f.write('0-1\torig-all-scqa: %s\n' % orig_scqa_folder + '/eval_predictions.json')
        for i, adv_type in enumerate(adv_names):
            base_folder = os.path.join(race_test_root_dir, adv_type, model_name, res_folders['base'])
            f.write('%s-0\t\%s-base: %s\n' % (str(i+1), adv_type, base_folder + '/eval_predictions.json'))
        for i, adv_type in enumerate(adv_names):
            scqa_folder = os.path.join(race_test_root_dir, adv_type, model_name, res_folders['scqa'])
            f.write('%s-1\t\%s-scqa: %s\n' % (str(i+1), adv_type, scqa_folder + '/eval_predictions.json'))
        f.write('\t'.join(['id'] + [str(idx) + '-0\t' + str(idx) + '-1' for idx in range(len(adv_names) + 1)]) + '\n')
        for id_ in good_case_ids:
            f.write('\t'.join([str(v) for v in [
                id_,
                orig_base_correct[id_][0],
                orig_scqa_correct[id_][0],
            ] + [adv_base_wrong[id_][0] for adv_base_wrong in adv_base_wrongs]
              + [adv_scqa_correct[id_][0] for adv_scqa_correct in adv_scqa_corrects]
            ]) + '\n')
