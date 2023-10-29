from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from os import listdir
import pandas as pd

folder = "C:/Users/chenrick/Desktop/Python projects/vs_code/ml-project/demo_train/"
images=[]
labels=[]

for f in listdir(folder):
    l = 0.0
    if f.startswith('dog'):
        l = 1.0
    images.append(f)
    labels.append(l)

df = pd.DataFrame({
    'images':images,
    'labels':labels
})
 
img_width = 128
img_height = 128
img_channels = 3

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, img_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_width, img_height, img_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu', input_shape=(img_width, img_height, img_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])


from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
df['labels'] = df['labels'].replace({0:'cat',1:'dog'})
training_set, validation_set = train_test_split(df, test_size=0.2)

training_set = training_set.reset_index(drop=True)
validation_set = validation_set.reset_index(drop=True)

total_train = training_set.shape[0]
total_validation = validation_set.shape[0]

batch_size = 15
training_generator = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
validation_generator = ImageDataGenerator(rescale=1./255)

training_flow = training_generator.flow_from_dataframe(training_set, folder, x_col='images', y_col='labels', target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size)
validation_flow = validation_generator.flow_from_dataframe(validation_set, folder, x_col='images', y_col='labels', target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size)

epochs = 10
history= model.fit_generator(training_flow, epochs=epochs, validation_data=validation_flow, validation_steps = total_validation//batch_size, steps_per_epoch=total_train//batch_size)

