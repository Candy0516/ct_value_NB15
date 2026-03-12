import os, sys, time
import configparser
from ct_value.f2_score import score
from ct_value.f3_putback import putback
from ct_value.f4_sum import sum
from ct_value.f5_sample import sample
from ct_value.f6_map_testset_fix import map_testset, map_valset
from ct_value.f7_sum_testest_fix import do_sum, do_sum_val

# === Logger 工具 ===
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class Logger(object):
    def __init__(self, filename='Log/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    @classmethod
    def timestamped_print(self, *args, **kwargs):
        _print(time.strftime("[%Y/%m/%d %X]"), *args, **kwargs)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def log_history(name_s_log):
    createFolder('Log/')
    sys.stdout = Logger('Log/' + name_s_log + '.log', sys.stdout)
    sys.stderr = Logger('Log/' + name_s_log + '.err', sys.stderr)

# === 設定讀取 ===
configreader = configparser.ConfigParser()
configreader.read('config.ini', encoding='utf-8')
config = dict(configreader.items('p-value'))
del configreader
config['label_column'] = list(config['label_column'].split(' '))
config['log'] = bool(config['log'])

# === 主流程 ===
if __name__=='__main__':
    if config['log']:
        _print = print
        print = Logger.timestamped_print
        log_history(os.path.basename(__file__))

    #all_labels = ['Dead within 24hr', 'Dead within 72hr', 'Dead within 168hr', 'Finally dead']
    #sample_types = ['none', 'under', 'over', 'smote', 'KMeansSMOTE_binary', 'classweight']


    # 設定資料根目錄
    BASE_DIR = r"D:\candy\NB15"


    '''
    for dead_label in range(len(all_labels)):
        print(f"\n{'='*60}")
        print(f"\033[35m[Label index {dead_label}] {all_labels[dead_label]}\033[0m")
    '''
    # for label in all_labels:
    #     print(f'label: {label}\n')

    #     for sample_type in sample_types:
    #         print(f"\n\033[35m[Sample Type] {sample_type}\033[0m")

            #label_folder = f'label_{dead_label}'
    preprocess_path = os.path.join(BASE_DIR, 'data', '1_preprocess')
    score_path = os.path.join(BASE_DIR, 'data', '2_score')
    pvalue_path = os.path.join(BASE_DIR, 'data', '3_DataWithPvalue')
    sum_path = os.path.join(BASE_DIR, 'data', '4_sum')
    test_map_path = os.path.join(BASE_DIR, 'data', '6_mapped_test')
    sum_test_path = os.path.join(BASE_DIR, 'data', '7_sum_test')
    val_map_path = os.path.join(BASE_DIR, 'data', '6_mapped_val')
    sum_val_path = os.path.join(BASE_DIR, 'data', '7_sum_val')


    os.makedirs(score_path, exist_ok=True)
    os.makedirs(pvalue_path, exist_ok=True)
    os.makedirs(sum_path, exist_ok=True)
    os.makedirs(test_map_path, exist_ok=True)
    os.makedirs(sum_test_path, exist_ok=True)
    os.makedirs(val_map_path, exist_ok=True)
    os.makedirs(sum_val_path, exist_ok=True)

    # === f2_score ===
    print('f2_score')
    file_path = os.path.join(preprocess_path, 'train.csv') #改成test
    score(file_path, score_path, config['label_column'], print)

    # === f3_putback ===
    print('f3_putback')
    file_path_data = os.path.join(preprocess_path, 'train.csv') #改成test
    read_path_pvalue = score_path
    putback(file_path_data, read_path_pvalue, pvalue_path, config['label_column'], print)

    # === f4_sum ===
    print('f4_sum')
    sum(pvalue_path, sum_path, config['label_column'], config['sum_or_count01'], print)

    #  f5_sample 
    # print('f5_sample')
    # file_path_ct = os.path.join('data', '4_sum', config['value'], 'train.csv')
    # file_path_ori = os.path.join('data', '1_preprocess', label_folder, sample_type, 'train.csv')
    # save_path = os.path.join('data', '5_sample', label_folder, sample_type)
    # os.makedirs(save_path, exist_ok=True)
    # sample(file_path_ct, file_path_ori, save_path, config['label_column'], config['sum_or_count01'], print)

    
    # === f6_map_testset ===
    print('f6_map_testset')
    file_path_test = os.path.join(preprocess_path, 'test.csv')
    file_path_val = os.path.join(preprocess_path, 'val.csv')
    map_table_path = os.path.join(score_path, 'statistic', 'map_table')
    map_testset(file_path_test, test_map_path, config['label_column'], map_table_path, print)
    map_valset(file_path_val, val_map_path, config['label_column'], map_table_path, print)

    # === f7_sum_testset ===
    print('f7_sum_testset')
    do_sum(test_map_path, sum_test_path, config['label_column'], print)
    do_sum_val(val_map_path, sum_val_path, config['label_column'], print)
                        