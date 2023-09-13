import logging

from pytorch_dl import *


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    clear_memory()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Constants
    labels_encoding = ['LD', '2lux', '5lux', 'LL'] # it can be somewhat confusing because I encoded it with 1,2,3,4 previously
    BATCH_SIZE = 20
    NUM_FILES_PER_SUBJECT = 10

    folder_path = filedialog.askdirectory()
    logging.info(f"The path: {folder_path} is going to be treated now.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using: {device} as processing device.")

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_encoding)

    # Initialize Data Loader
    data_loader = CustomDataLoaderWithSubjects(folder_path, num_files_per_subject=NUM_FILES_PER_SUBJECT)

    # Results directory
    folder_name = os.path.basename(folder_path)
    parent_directory = os.path.dirname(folder_path)
    result_dir = os.path.join(parent_directory, f"result_{folder_name}")
    os.makedirs(result_dir, exist_ok=True)

    # model = MFCC_CNN()
    model = MFCC_CNN()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.00001)  # 0.0002 lr
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8)

    best_fold_model_state = None
    best_fold_accuracy = 0
    # Cross-Validation Loop
    for fold in range(data_loader.num_folds):
        logging.info(f"Starting Fold {fold + 1}")
        # Split data for the current fold
        data_loader.split_data_files()

        # Normalize the training set and use its stats for the validation and test sets
        train_dataset = CustomDataset(data_loader.train_files, labels=label_encoder)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
        train_mean, train_std = compute_mean_std(train_loader)
        logging.info(f"Training set has been normalized")
        directory = r"C:\Users\yfant\OneDrive\Desktop"

        # Load datasets with normalization
        train_dataset = CustomDataset(data_loader.train_files, labels=label_encoder, mean=train_mean, std=train_std)
        val_dataset = CustomDataset(data_loader.val_files, labels=label_encoder, mean=train_mean, std=train_std)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
        logging.info('Data has been loaded and ready to be processed.')

        trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, loss_fn, device)
        trainer.train(n_epochs=5, best_accuracy=95, batch_size=BATCH_SIZE, num_files=NUM_FILES_PER_SUBJECT,
                      folder_name=os.path.basename(folder_path), directory=directory)

        if max(trainer.val_accuracies) > best_fold_accuracy:
            best_fold_accuracy = max(trainer.val_accuracies)
            best_fold_model_state = trainer.get_best_model_state()

        # Move to the next fold
        data_loader.next_fold()
    logging.info(f"All {data_loader.num_folds} folds have been processed.")
    logging.info("Let's start the final test")
    model.load_state_dict(best_fold_model_state)
    test_dataset = CustomDataset(data_loader.test_files, labels=label_encoder, mean=train_mean, std=train_std)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_accuracy, _, _ = trainer.evaluate(test_loader)
    logging.info(f"Final test accuracy using best fold: {test_accuracy:.2f}%")
