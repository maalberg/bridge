import os
import sys
import importlib.util
import torch

from pytorch_nndct.apis import torch_quantizer
from util import normalize_test, read_file


def main(model_moduledir, model_classname):

    model_modulename = 'model'
    model_filename = 'model.py'
    model_tester_classname = f'{model_classname}_tester'
    dataset_builder_classname = f'{model_classname}_dataset_builder'
    print(f"[note]: model is located in {model_moduledir}")
    print(f"[note]: model tester is {model_tester_classname}")
    print(f"[note]: dataset builder is {dataset_builder_classname}")


    # --!
    # --!----------- dynamically load model classes -----------!
    # --!

    model_modulepath = os.path.join(model_moduledir, model_filename)
    spec = importlib.util.spec_from_file_location(model_modulename, model_modulepath)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model_class = getattr(model_module, model_classname)
    model_tester_class = getattr(model_module, model_tester_classname)
    model_params = getattr(model_module, 'model_params')
    dataset_builder_class = getattr(model_module, dataset_builder_classname)
    data_params = getattr(model_module, 'data_params')
    dataset_params = getattr(model_module, 'dataset_params')



    # --!
    # --!----------- instantiate model -----------!
    # --!

    weights_path = os.path.join(model_moduledir, f'{model_classname}.pth')

    model = model_class()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    model_tester = model_tester_class()


    # --!
    # --!----------- load normalization constants -----------!
    # --!

    norms_path = os.path.join(model_moduledir, f'{model_classname}_norms')

    # --! there are two constants that we read: mean and standard deviation
    nnorm = 2

    norms = read_file(norms_path, nnorm)
    print(f'[note]: read normalization constants with shape {norms.shape} from file {norms_path}')

    # --! extract data mean and standard deviation as lists
    norms = torch.squeeze(norms, 0)
    data_mean = norms[0].tolist()
    data_std = norms[1].tolist()


    # --!
    # --!----------- create data loader -----------!
    # --!

    data_path = os.path.join(model_moduledir, f'{model_classname}_data')
    labels_path = os.path.join(model_moduledir, f'{model_classname}_labels')

    data = read_file(data_path, data_params[model_classname]['data_nsample'])
    labels = read_file(labels_path, data_params[model_classname]['labels_nsample'])
    print(f'[note]: read data shape is {data.shape}')
    print(f'[note]: read labels shape is {labels.shape}')

    data = normalize_test(data, data_mean, data_std)
    labels = torch.squeeze(labels, -1)

    dataset = (dataset_builder_class()
        .add_labels(labels)
        .build(data, **dataset_params[model_classname]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f'[note]: dataset item shape is {dataset[0][0].shape}')
    print(f'[note]: number of items in dataset is {len(dataset)}')


    # --!
    # --!----------- create quantizer in test mode -----------!
    # --!

    # --! determine shape of model input
    if model_params[model_classname]['input_flattened']:
        quantizeri_dummy = torch.flatten(torch.unsqueeze(dataset[0][0], 0), start_dim=1)
    else:
        if model_params[model_classname]['input_transposed']:
            quantizeri_dummy = torch.unsqueeze(torch.transpose(dataset[0][0], -1, -2), 0)
        else:
            quantizeri_dummy = torch.unsqueeze(dataset[0][0], 0)
    print(f'[note]: quantizer dummy input shape is {quantizeri_dummy.shape}')

    quantizer = torch_quantizer(
        'test',
        model,
        (quantizeri_dummy,))
    quant_model = quantizer.quant_model
    quant_model.eval()


    # --!
    # --!----------- test quantized model -----------!
    # --!

    print(f"[note]: testing quantized model {model_classname}...")

    model_tester(model, dataloader, quant_model)


    # --!
    # --!----------- export quantized model -----------!
    # --!
    
    quantizer.export_xmodel()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"[usage]: python3 {sys.argv[0]} <modeldir> <classname>")
    else:
        main(sys.argv[1], sys.argv[2])

