import torch
from matplotlib import pyplot as plt


class HistoryWriter():
    def __init__(self):
        self.history = {}

    def save_record(self, key, value):
        if key not in self.history:
            self.history[key] = []

        # сразу сохраняем CPU-float
        if torch.is_tensor(value):
            value = value.detach().cpu().item()

        self.history[key].append(value)

    def plot_history(self, x_label, save_path):
        for key, values in self.history.items():
            fig, ax = plt.subplots()
            clean_values = [
                v.detach().cpu().item() if torch.is_tensor(v) else float(v)
                for v in values
            ]

            ax.plot(clean_values)
            ax.set_xlabel(x_label)
            ax.set_ylabel(key)
            fig.savefig(f"{save_path}_{key}.png")
            plt.close(fig)
