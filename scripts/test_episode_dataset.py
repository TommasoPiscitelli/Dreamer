from torch.utils.data import DataLoader

from dreamer_carracing.data.episode_dataset import CarRacingEpisodeDataset


def main():
    dataset = CarRacingEpisodeDataset(
        data_dir="data/raw/train",
        seq_len=50,
        image_size=64,
    )

    print("num subsequences:", len(dataset))

    item = dataset[0]
    print("single obs:", item["obs"].shape)
    print("single actions:", item["actions"].shape)
    print("single rewards:", item["rewards"].shape)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print("batch obs:", batch["obs"].shape)
    print("batch actions:", batch["actions"].shape)
    print("batch rewards:", batch["rewards"].shape)


if __name__ == "__main__":
    main()
