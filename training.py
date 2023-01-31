import torch
import src

from src.utils.dataset import load_data

if __name__=="__main__":
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' 
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') 

    model = src.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=1,
        activation=ACTIVATION,
    )

    loss = src.utils.base.SumOfLosses(
        src.utils.losses.DiceLoss(),
        src.utils.losses.BCELoss()
    )

    metrics = [
        src.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = src.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = src.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    train_loader, valid_loader = load_data(test_size=0.3, batch_size=1, img_size=256, dir='./data/', artificial_increase=20)

    max_score = 5
    trash = 0
    for i in range(0, 10):
        if trash > 6:
            break
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        if max_score > valid_logs['dice_loss + bce_loss']:
            max_score = valid_logs['dice_loss + bce_loss']
            torch.save(model, './checkpoint/best_model.pth')
            trash = 0
            print('Model saved!')
        else:
            trash +=1
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.56
            print('Decrease decoder learning rate to 1e-5!')