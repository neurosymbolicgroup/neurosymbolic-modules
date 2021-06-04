import os
from collections import namedtuple

def mean(x):
    return sum(x) / len(x)


def read_lines(path):
    with open(path, 'r') as f:
        lines = [l[:-1] for l in f]
    return lines


def get_epoch(line):
    words = line.split()
    second_word = words[1]
    return int(second_word)


def get_accuracy(line):
    words = line.split()
    acc_ix = words.index('acc:')
    acc_word = words[acc_ix + 1]
    return float(acc_word)


def is_completed(lines):
    if any(['epoch: 9900' in l for l in lines]):
        if not 'with name model' in lines[-1]:
            print(lines)
            assert False
        return True
    return False


def get_model_id(lines):
    last_line = lines[-1]
    model_id = last_line[:last_line.index(' ')]
    return model_id


def analyze_pg_file(lines):
    model_id = get_model_id(lines)

    epoch_lines = [l for l in lines if l.startswith('epoch: ')]
    last_5 = [l for l in epoch_lines if get_epoch(l) in [9500, 9600, 9700, 9800, 9900]]
    assert len(last_5) == 5

    last_5_accs = [get_accuracy(l) for l in last_5]
    acc_mean = round(mean(last_5_accs), 4)

    return model_id, acc_mean


def get_paths(source_dir):
    paths = os.listdir(source_dir)
    non_hidden_paths = [p for p in paths if p[0] != '.']
    return non_hidden_paths


def analyze_all_files(source_dir):
    paths = get_paths(source_dir)
    # for official experiments
    paths = [path for path in paths if '24_official' in path]

    count = 0
    results = {}
    for path in paths:
        lines = read_lines(source_dir + '/' + path)
        if is_completed(lines):
            model_id, acc_mean = analyze_pg_file(lines)
            results[path] = model_id, acc_mean
            count += 1

    print(f'Analyzed {count} files successfully')
    # assert 'fw_d2_1.out' in results.keys()
    for (path, (model_id, acc_mean)) in sorted(results.items()):
        print(f"path: {path}")
        print(f"id: {model_id}\tacc: {100*acc_mean:.2f}")


def get_24_policy_gradient_models(src_dir='out/') -> List[Tuple[str, str]]:
    paths = os.listdir(src_dir)

    out = []
    for path in paths:
        lines = data_analytics.read_lines(src_dir + '/' + path)
        assert data_analytics.is_completed(lines), f"path {path} not completed"
        if data_analytics.is_completed(lines):  # and path.endswith('_1.out'):
            model_id = data_analytics.get_model_id(lines)
            model = utils.load_mlflow_model(model_id)
            forward_only = 'bidir' not in path
            if forward_only:
                if len(model.ops) != 4:
                    print(f"path {path} not fw={forward_only}")
                    continue
            else:
                if len(model.ops) != 12:
                    print(f"path {path} not fw={forward_only}")
                    continue

            out.append((model_id, path))

    return sorted(out, key=lambda t: t[1])


if __name__ == '__main__':
    analyze_all_files('out/')
