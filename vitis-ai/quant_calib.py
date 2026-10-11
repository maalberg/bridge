import os
import sys
import importlib.util
import torch

from pytorch_nndct.apis import torch_quantizer
from util import normalize_calib, read_file, write_file


def main(model_moduledir, model_classname, data_nsample):

    model_modulename = 'model'
    model_filename = 'model.py'
    dataset_classname = f'{model_classname}_dataset'
    print(f"[note]: model is located in {model_moduledir}")
    print(f"[note]: calibrating quantized model {model_classname}")
    print(f"[note]: dataset is {dataset_classname}")


    # --!
    # --!----------- dynamically load model classes -----------!
    # --!

    model_modulepath = os.path.join(model_moduledir, model_filename)
    spec = importlib.util.spec_from_file_location(model_modulename, model_modulepath)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model_class = getattr(model_module, model_classname)
    dataset_class = getattr(model_module, dataset_classname)


    # --!
    # --!----------- instantiate model -----------!
    # --!

    weights_path = os.path.join(model_moduledir, f'{model_classname}.pth')

    model = model_class()
    model.load_state_dict(torch.load(weights_path))
    model.eval()


    # --!
    # --!----------- create data loader -----------!
    # --!

    data_path = os.path.join(model_moduledir, f'{model_classname}_data')
    labels_path = os.path.join(model_moduledir, f'{model_classname}_labels')

    data = read_file(data_path, data_nsample)
    labels = read_file(labels_path, data_nsample)

    data, data_mean, data_std = normalize_calib(data)
    labels = torch.squeeze(labels, -1)
    print(f'[note]: read data shape is {data.shape}')
    print(f'[note]: read labels shape is {labels.shape}')

    dataset = dataset_class(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f'[note]: dataset item shape is {dataset[0][0].shape}')


    # --!
    # --!----------- save normalization constants -----------!
    # --!

    norms_path = os.path.join(model_moduledir, f'{model_classname}_norms')

    norms = torch.cat([data_mean, data_std], dim=0).unsqueeze(0)
    write_file(norms_path, norms)
    print(f'[note]: saved normalization constants with shape {norms.shape} to file {norms_path}')


    # --!
    # --!----------- create quantizer in calibration mode -----------!
    # --!

    quantizeri_dummy = torch.unsqueeze(torch.transpose(dataset[0][0], -1, -2), 0)
    print(f'[note]: quantizer dummy input shape is {quantizeri_dummy.shape}')

    quantizer = torch_quantizer(
        'calib',
        model,
        (quantizeri_dummy,))
    quant_model = quantizer.quant_model
    quant_model.eval()


    # --!
    # --!----------- exercise calibration mechanism -----------!
    # --!

    with torch.no_grad():
        for x, _ in dataloader:
            quant_model(torch.transpose(x, -1, -2))


    # --!
    # --!----------- export quantization config -----------!
    # --!

    quantizer.export_quant_config()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"[usage]: python3 {sys.argv[0]} <modeldir> <classname> <data_nsample>")
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

