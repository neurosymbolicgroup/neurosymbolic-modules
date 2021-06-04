from typing import List
from matplotlib import pyplot as plt


def analyze_data(path) -> List[int]:
    with open(path, 'r') as f:
        lines = [line[:-1] for line in f]
        solved_lines = [line for line in lines if line.startswith('solved')]

        def second_word(line):
            return line.split()[1]

        solved_counts = [int(second_word(line)) for line in solved_lines]
        return solved_counts


def get_fw_data(path='out/double_and_add_fw_only.out') -> List[int]:
    return analyze_data(path)


def get_bidir_data(path='out/double_and_add_bidir.out') -> List[int]:
    return analyze_data(path)


def plot():
    # Create data
    # fw = [0, 0, 3, 5, 28, 37, 68, 75, 169, 142, 143, 187, 178, 247, 209, 185, 164, 285, 233, 291, 266, 332, 319, 206, 240, 349, 332, 350, 366, 324, 349, 325, 356, 328, 335, 307, 412, 348, 313, 369, 320, 279, 323, 346, 387, 368, 366, 369, 251, 343]
    # bidir = [500] * len(fw)
    # bidir[0] = 0
    fw = get_fw_data()
    bidir = get_bidir_data()
    assert len(bidir) == 2
    assert len(fw) == 50
    continued_bidir = [500] * len(fw)
    continued_bidir[0] = 0
    assert continued_bidir[0:2] == bidir
    bidir = continued_bidir

    fw = [100 * x / 500 for x in fw]
    bidir = [100 * x / 500 for x in bidir]
    x = range(len(fw))

    # Area plot
    plt.figure(figsize=(6, 3))
    plt.fill_between(x, bidir, color='lightcoral', alpha=0.2)
    plt.plot(x, bidir, color='indianred', alpha=1, label='bidirectional')

    plt.fill_between(x, fw, color='skyblue', alpha=1)
    plt.plot(x, fw, color="dodgerblue", alpha=1, label='forward-only')
    plt.xticks(range(0, len(fw), 10))
    plt.xlabel('Epoch')
    plt.ylabel('Percent of tasks solved')
    plt.grid(axis='y')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=.05)

    # Show the graph
    plt.show()

    # Note that we could also use the stackplot function
    # but fill_between is more convenient for future customization.
    # plt.stackplot(x,y)
    # plt.show()


if __name__ == '__main__':
    plot()
