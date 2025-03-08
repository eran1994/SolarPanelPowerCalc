from ultralytics import YOLO



def get_yolo_params(name, epochs=300, project='models', weights='yolov5m',
                    evolve=0, device='cuda:0'):
    yolo_parameters = {
        'data': 'sp_dataset.yaml',
        # 'weights': f'{project}/{weights}/weights/best.pt',
        'weights': weights + '.pt' if not weights.endswith('.pt') else '',
        'imgsz': 256,
        'batch_size': 16,
        'workers': 4,
        'project': project,
        'name': name,
        'epochs': epochs,
        'device': device
    }
    if evolve:
        yolo_parameters['evolve'] = evolve
        yolo_parameters['name'] += '_hyp'
    return yolo_parameters


def main():
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #epochs = 300
    #model = 'yolov5m'
    #model_name = f'{model}_{epochs}_test'

    model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(
        data="sp_dataset.yaml",
        epochs=1200,
        imgsz=256,#640 crush computer
        augment=True,
        optimizer='AdamW',
        lr0=0.005,  # Higher initial learning rate to break plateau
        lrf=0.001,  # Lower final learning rate to fine-tune toward the end
        momentum=0.85  # Slightly lower momentum for more stable updates
    )

    metrics = model.val()  # evaluate model performance on the validation set
    path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    main()
