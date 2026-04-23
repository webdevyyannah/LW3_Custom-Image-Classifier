# Laboratory Work 3 — Building a Custom Image Classifier with TensorFlow

## Results

### Model Performance Comparison

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Epochs | 10 | 15 |
| Training Accuracy | 100% | 77.39% |
| Validation Accuracy | 96.10% | 87.20% |
| Training Loss | ~0.0001 | 0.7414 |
| Validation Loss | 0.2866 | 0.4621 |
| Data Augmentation | None | RandomFlip, RandomRotation, RandomZoom |
| Dropout | None | Dropout(0.3) x2 |
| Overfitting | Yes | No |

---

## Guide Questions

### Part 1 — Dataset Preparation

**How did you organize your dataset in Google Drive?**

I organized my dataset in Google Drive by creating a main folder called `IMAGE DATA SET` inside MyDrive. Inside that folder, I created subfolders for each plant species category, and each subfolder contains at least 250 images of that specific plant. The folder names were used automatically as the class labels by TensorFlow.

**Why is folder structure important for TensorFlow image loading?**

The folder structure is important because `tf.keras.utils.image_dataset_from_directory()` uses the subfolder names as the class labels. If the folders are not properly organized, TensorFlow won't be able to correctly assign labels to the images during training, which would result in incorrect classifications.

---

### Part 2 — Model Training

**What is the role of convolutional layers in image classification?**

Convolutional layers are responsible for automatically learning and extracting features from images, such as edges, textures, shapes, and patterns. Each layer learns increasingly complex features — the first layers detect simple edges, while deeper layers recognize more complex structures. This allows my model to distinguish between different plant species based on their visual characteristics.

**Why do we split data into training and validation sets?**

We split the data into training and validation sets so that we can evaluate how well the model generalizes to data it has never seen before. If we only trained on all the data without a separate validation set, we wouldn't know if the model is actually learning or just memorizing the training images. I used an 80/20 split — 80% for training and 20% for validation.

---

### Part 3 — Performance Analysis

**What accuracy did your model achieve?**

My first model achieved a validation accuracy of **96.10%** after 10 epochs of training. The training accuracy reached 100% by the final epoch, while the validation accuracy stabilized around 96%.

**How did the number of images affect the model's performance?**

Having at least 250 images per category gave the model enough examples to learn the distinguishing features of each plant species. With more images, the model is exposed to more variety and visual differences, which helps it generalize better. If there were fewer images, the model would likely overfit much faster and perform poorly on new data.

---

### Part 4 — Critical Thinking

**What challenges did you encounter while using your own dataset?**

One challenge I encountered was that my first model showed signs of overfitting — the training accuracy reached 100% while the validation accuracy was lower and the validation loss was increasing. This means the model was memorizing the training images rather than truly learning to classify them. Another challenge was ensuring that each plant category had enough quality images so that the model could learn properly.

**How can data augmentation improve your model?**

Data augmentation improves my model by artificially increasing the variety of training data. By randomly flipping, rotating, and zooming the images, the model sees many different versions of the same plant, which helps it become more robust and generalize better to images it hasn't seen before. This reduces overfitting because the model can no longer memorize exact images.

---

### Part 5 — Application

**Suggest a real-world application for your trained model.**

My trained model could be used as a plant identification app for farmers, botanists, or students in agriculture. Users could simply take a photo of a plant and the app would identify the species instantly. This could be especially useful in the Philippines for identifying local plant species and medicinal herbs.

**How can this system be integrated into a mobile or web application?**

The trained model can be saved using `model.save()` and then deployed using TensorFlow Lite for mobile apps or TensorFlow.js for web apps. A mobile app could allow users to take a photo directly from their camera, send it to the model for prediction, and display the plant name along with the confidence score — just like what my model does, which predicted "TALISAY TREE" with 99.9% confidence.

---

### Activity 3A — Improving and Evaluating a Custom Image Classifier

**What signs indicated overfitting in your first model?**

The clearest sign of overfitting in my first model was the growing gap between training accuracy and validation accuracy. By epoch 8, my training accuracy reached 100% and the training loss dropped to nearly 0, while the validation accuracy stayed around 96% and the validation loss remained at around 0.28. The accuracy graph also showed the training line shooting up steeply while the validation line flattened out — a classic overfitting pattern.

**How did data augmentation affect validation accuracy?**

After applying data augmentation, the validation accuracy in my improved model actually became higher than the training accuracy, which is a sign that the model is generalizing well instead of memorizing. The validation accuracy steadily improved over all 15 epochs, reaching 87.2%. While this number appears lower than the first model's 96.1%, the behavior of the model is much healthier because there is no overfitting.

**What is the purpose of dropout layers?**

Dropout layers randomly deactivate a percentage of neurons during training — in my model, I used a dropout rate of 0.3, meaning 30% of neurons are randomly turned off each step. This prevents the model from becoming too reliant on specific neurons and forces it to learn more distributed and robust features. The result is a model that generalizes better to new, unseen images.

**Why does data augmentation improve generalization?**

Data augmentation improves generalization because it exposes the model to more varied versions of the training images — flipped, rotated, and zoomed. This means the model learns to recognize plant species regardless of orientation or scale, rather than just memorizing the exact training images. It effectively makes my dataset larger and more diverse without needing to collect new photos.

**Compare accuracy before and after improvements.**

My first basic model achieved a validation accuracy of **96.10%** but showed clear overfitting, with training accuracy hitting 100% and validation loss staying high. After adding data augmentation and dropout, my improved model achieved **87.2% validation accuracy**, but with much healthier training behavior — both accuracy curves improved together, and the validation accuracy was consistently higher than the training accuracy.

**Which technique contributed most to improvement?**

In my case, data augmentation contributed the most to improvement because it addressed the root cause of overfitting — the model seeing the same images repeatedly. By generating varied versions of the training images, the model was forced to learn real features instead of memorizing patterns. Dropout also helped by preventing over-reliance on specific neurons, but augmentation had the bigger visible impact on the loss and accuracy curves.

**Why is saving the model important?**

Saving the model is important because training a CNN from scratch is time-consuming and resource-intensive. By saving the trained model using `model.save()`, I can reuse it later for predictions without retraining. It also makes it possible to deploy the model in applications — for example, loading it in a mobile or web app to classify plant images in real time.

**How can this model be deployed in a real-world system?**

My saved model (stored as `my_image_classifier.keras`) can be deployed in several ways. For a web application, it can be converted and served using TensorFlow.js or a Flask/FastAPI backend. For a mobile app, it can be converted to TensorFlow Lite format, which is optimized for mobile devices. Users would upload or capture a photo, and the system would return the predicted plant species along with its confidence score — similar to how my model predicted "TALISAY TREE" with 99.9% confidence during testing.

---

## Conclusion

This laboratory activity demonstrated the full process of building, evaluating, and improving a custom image classifier using TensorFlow and a personal plant species dataset. My baseline model achieved a high validation accuracy of 96.10%, but the training graphs revealed a clear overfitting problem — the model was memorizing the training data rather than truly learning to classify plants.

By applying data augmentation and dropout regularization in my improved model, I was able to address overfitting and produce a model with much healthier training behavior. Although the final validation accuracy of 87.2% appears lower than the baseline, the improved model generalizes far better to unseen images, as proven by its ability to correctly predict "TALISAY TREE" with 99.9% confidence on a real test image.

This activity reinforced the importance of not just chasing high accuracy numbers, but also understanding how a model behaves during training. A model that learns is always more valuable than a model that memorizes. Moving forward, this classifier could be further improved with more training data, transfer learning using pre-trained models like MobileNet or EfficientNet, and eventually deployed as a mobile or web-based plant identification tool.

## 🔗 Project Links

- 📓 **Google Colab Notebook:** [(https://colab.research.google.com/drive/1ZAjxu45z3iDFpgQRa_puX-c4m5VQesAS?usp=sharing)]
- 📁 **Google Drive Dataset:** [(https://drive.google.com/drive/folders/1TRQJ9ZjW8XNAK6L1VdbcqLDDwcdhuwcO?usp=sharing)]
- 🧠 **Saved Model:** [(https://drive.google.com/file/d/19L1TODQCLFHRFOioXjQewesOzPX2qbG1/view?usp=drive_link)]
