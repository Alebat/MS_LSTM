import numpy as np
import os
import random
import torch
import torch.utils.data as data


def default_loader(path):
    #feature = np.reshape(np.fromfile(path, dtype="float32"), [-1, 2048])
    feature = np.load(path)
    # return np.amax(feature, axis=0)
    return feature
    # return Image.open(path).convert('RGB')


def transform(fea, max_frames):
    num_frames, _ = fea.shape
    if num_frames > max_frames:
        start_idx = random.choice(range(num_frames - max_frames))
        new_fea = fea[start_idx:start_idx + max_frames, :]
    else:
        new_fea = np.zeros([max_frames, fea.shape[1]])
        new_fea[0:num_frames, :] = fea
    return torch.Tensor(new_fea)


class videoDataset(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None,
                 suffixes=".binary", loader=default_loader, data=None):
        if data is not None:
            videos = data
        else:
            if isinstance(suffixes, str):
                suffixes = [suffixes]
            with open(label) as fh:
                videos = []
                for line in fh.readlines():
                    video_id, mark = line.strip().split(',')
                    mark = float(mark)
                    for suffix in suffixes:
                        videos.append((f'{video_id}{suffix}', mark))
        self.root = root
        self.videos = videos
        self.transform = lambda x: transform(x, 300)
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, score = self.videos[index]
        fea = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            fea = self.transform(fea)
        return fea, torch.Tensor([score])
        # return fea

    def __len__(self):
        return len(self.videos)


if __name__ == "__main__":
    # torch.utils.data.DataLoader
    dataset = videoDataset(root="runs/extract_c3d_fc6",
                           label="./data/train_dataset.txt")
    videoLoader = torch.utils.data.DataLoader(dataset,
                                              batch_size=16, shuffle=True, num_workers=0)

    for i, (features, scores) in enumerate(videoLoader):
        import pdb
        pdb.set_trace()
        print((i, len(features)))
