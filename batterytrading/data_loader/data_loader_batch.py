from batterytrading.data_loader import get_data
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np


class Data_Loader_batch(DataLoader):
    def __init__(
        self,
        batch_size: int,
        price_time_horizon: float = 1,
        share_of_samples_start: float = 0.0,
        share_of_samples_end: float = 0.7,
    ):
        """
        Initialize the Data Loader
        Args:
            price_time_horizon: Number of days to return intraday prices
        """
        assert share_of_samples_start >= 0.0
        assert share_of_samples_end <= 1.0
        assert share_of_samples_start < share_of_samples_end

        self.batch_size = batch_size
        self.data = get_data()
        self.data = self.data["intraday_15min"].sort_index().values
        self.data = self.data[int(share_of_samples_start * len(self.data)) : int(share_of_samples_end * len(self.data))]
        print(len(self.data))
        self.current_index = 0
        self.has_next = True
        self.n_timesteps = int(price_time_horizon * 24 * 4)
        super(Data_Loader_batch, self).__init__(self.data, batch_size=self.batch_size, shuffle=False)
        print("asf")

    def __next__(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # torch.Tensor, torch.Tensor
        if self.current_index + self.batch_size + self.n_timesteps < len(self.data):
            start_index = [self.current_index + i for i in range(self.batch_size)]

            x = [self.data[i : i + self.n_timesteps] for i in start_index]
            y = [self.data[i + self.n_timesteps : i + self.n_timesteps * 2] for i in start_index]
            x = np.vstack(x)
            y = np.vstack(y)
            self.current_index += 1  # self.batch_size

            return torch.tensor(x, dtype=torch.float32).reshape(self.batch_size, self.n_timesteps, -1), torch.tensor(y, dtype=torch.float32).reshape(
                self.batch_size, self.n_timesteps, -1
            )
        else:
            self.has_next = False
            raise StopIteration()

    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__next__()

    def _get_iterator(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__next__()


if __name__ == "__main__":
    data_loader = Data_Loader_batch(16, price_time_horizon=1.5, share_of_samples_start=0.0, share_of_samples_end=0.7)
    while data_loader.has_next:
        x, y = next(data_loader)
        print(x, y)
