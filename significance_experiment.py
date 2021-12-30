
import numpy
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Pixel Significance Experiment for "No-Box" Neural Network Attacks

# This script guides a simple experiment which demonstrates how the underlying statistics
# of an input space (namely, the "significance" of input neurons) can form the basis
# for establishing expectations for what weights a neural network might develop while
# training on samples from the input space. This technique allows for the creation of
# promising neural network attacks without any knowledge about the target network, even
# attacks for a network that doesn't yet exist!

# Nick Kantack 12-1-2021

# To use saved pixel signficances rather than recalculating them, leave this flag True
LOAD_SIGNIFICANCES = True

# To save the average weights calculated during the experiment, set this flag True
SAVE_AVERAGE_WEIGHTS = False

# If you want to just see the results from a past experiment (without running one yourself)
# set this flag to False
GENERATE_FRESH_AVERAGE_WEIGHTS = True

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

IMAGE_DIM = 28
IMAGE_DIM_SQUARED = IMAGE_DIM**2

def main():

	(x_train, y_train), _ = keras.datasets.mnist.load_data()
	
	# There are 10 digits that are represented in the dataset
	num_classes = 10
	
	# vectorize x_train
	x_train = x_train.reshape(-1, IMAGE_DIM_SQUARED)
	x_train = x_train.astype('float32') / 255
	classes = y_train
	
	# each y element (the classification) needs to be remade as a vector with a 1 hot activation
	y_train = keras.utils.to_categorical(y_train, num_classes)
	
	# Let's get significance
	pixel_significance_by_class = 0
	if LOAD_SIGNIFICANCES:
		with open("pixel_significance.npy", "rb") as file:
			pixel_significance_by_class = numpy.load(file)
	else:
		pixel_significance_by_class = get_pixel_significance(x_train, classes, num_classes)
	
	plot_significance(pixel_significance_by_class, title="Pixel significances by class")
	
	if GENERATED_FRESH_AVERAGE_WEIGHTS:
		# Now see if randomly generated weights exhibit the statistics we expect
		produce_many_neural_networks(x_train, y_train, num_classes)
	else:
		plot_significance(numpy.load("mean_weights.npy"), title="Average weights after training (past experiment)")

def produce_many_neural_networks(x_train, y_train, num_classes):
	
	mean_weights = numpy.zeros((num_classes, IMAGE_DIM_SQUARED))
	
	sample_num = 20
	# Sample many neural networks
	for trial in range(sample_num):

		# Initialize a model with weights that are normally distributed and zero mean
		model = keras.Sequential([
				keras.Input(shape=(IMAGE_DIM_SQUARED,)),
				layers.Dense(num_classes, activation="sigmoid", kernel_initializer="RandomNormal")])
		model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
		
		# Train the network
		batch_size = 64
		epochs = 10
		model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

		# Extract the weights
		weights, _ = model.layers[0].get_weights()
		if numpy.isnan(numpy.sum(weights)):
			print("Skipping nan weights - possible exploded gradient")
			print(weights)
		else:
			mean_weights += weights.T
		print(f"Finished with trial {trial + 1}/{sample_num}")

	# Save and plot results
	if SAVE_AVERAGE_WEIGHTS:
		numpy.save("mean_weights.npy", mean_weights)
	plot_significance(mean_weights / sample_num, title="Average weights after training")

def plot_significance(significance_data, title=None):
	fig, axs = plt.subplots(2, 5, sharey=True)
	for i in range(10):
		row = int(i / 5)
		column = i % 5
		axs[row, column].imshow(numpy.reshape(significance_data[i,:], (IMAGE_DIM,IMAGE_DIM)))
		axs[row, column].set_title(i)
		axs[row, column].set_xticks([])
		axs[row, column].set_yticks([])
	fig.tight_layout()
	if title is not None:
		fig.suptitle(title)
	plt.show()
	
def get_pixel_significance(data, classes, class_count):

	pixel_count = data.shape[1]
	sample_count = data.shape[0]
	
	# To calculate pixel significance, it is helpful to first calculate pixel means by class
	pixel_mean_by_class = numpy.zeros((class_count, pixel_count))
	sample_count_by_class = numpy.zeros((class_count,))
	pixel_significance_by_class = numpy.zeros((class_count, pixel_count))
	for pixel_index in range(pixel_count):
		for sample_index in range(sample_count):
			sample_class = classes[sample_index]
			pixel_mean_by_class[sample_class, pixel_index] += data[sample_index, pixel_index]
			sample_count_by_class[sample_class] += 1
		print(f"Count finished for pixel index {pixel_index}/{pixel_count}")
			
	for class_index in range(class_count):
		pixel_mean_by_class[class_index, :] /= sample_count_by_class[class_index]	
	
	# Now calculate significance
	for pixel_index in range(pixel_count):
		for class_index in range(class_count):
			pixel_significance_by_class[class_index, pixel_index] = pixel_mean_by_class[class_index, pixel_index] - numpy.mean(pixel_mean_by_class[:,pixel_index])
			
	numpy.save(f"pixel_significance.npy", pixel_significance_by_class)
	return pixel_significance_by_class

if __name__ == "__main__":
	main()