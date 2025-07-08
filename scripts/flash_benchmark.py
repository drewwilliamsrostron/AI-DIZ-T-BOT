import torch
from torch.profiler import profile, schedule, tensorboard_trace_handler
from artibot.dataset import HourlyDataset
from artibot.ensemble import TradingModel


def benchmark(model, loader, label):
    with profile(
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(f"./logs/{label}"),
        record_shapes=True,
    ) as prof:
        for batch in loader:
            model(**batch)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))


if __name__ == "__main__":
    ds = HourlyDataset("data.csv")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, pin_memory=True, num_workers=4
    )
    model = TradingModel().cuda()
    torch.backends.cuda.enable_flash_sdp(False)
    benchmark(model, loader, "default")
    torch.backends.cuda.enable_flash_sdp(True)
    benchmark(model, loader, "flash_sdp")
