import tensorflow as tf

def get_train_val():

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/Train",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224,224),
        batch_size=32,
        label_mode="categorical"
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/Train",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224,224),
        batch_size=32,
        label_mode="categorical"
    )

    return train_data, val_data

def get_test():

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/Test",
        image_size=(224,224),
        batch_size=1,
        shuffle=False,
        label_mode="categorical"
    )

    return test_data