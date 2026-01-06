from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
training_dir = "horse-or-human/training"
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300,300),
    class_mode='binary'
)
