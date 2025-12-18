from matplotlib import pyplot as plt


class HistoryWriter():
    def __init__(self):
        self.history = {}

    def save_record(self, key, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    def plot_history(self, x_label, save_path):
        for key, values in self.history.items():
            fig, ax = plt.subplots()
            ax.plot(values)
            ax.set_xlabel(x_label)
            ax.set_ylabel(key)
            fig.savefig(f"{save_path}_{key}.png")
            plt.close(fig)
