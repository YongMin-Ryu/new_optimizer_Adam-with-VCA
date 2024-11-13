# Overview
## XAI

The XAI code used in this study is a Layer-wise Relevance Propagation (LRP) code consisting of a total of five hidden layers.

All codes are written in Python, and the parts written in Korean are used to refer to file names during the file download process.


As mentioned earlier, the hidden layer was modified to 5 to utilize XAI.

Therefore, the following code was used to check the weights propagated from each side.

    weights = model.get_weights()

    layer1_model = tf.keras.Model(inputs = model.input, outputs = model.layers[0].output)
    layer1_model_output = layer1_model.predict(input)

    layer2_model = tf.keras.Model(inputs = model.input, outputs = model.layers[1].output)
    layer2_model_output = layer2_model.predict(input)

    layer3_model = tf.keras.Model(inputs = model.input, outputs = model.layers[2].output)
    layer3_model_output = layer3_model.predict(input)

    layer4_model = tf.keras.Model(inputs = model.input, outputs = model.layers[3].output)
    layer4_model_output = layer4_model.predict(input)

    layer5_model = tf.keras.Model(inputs = model.input, outputs = model.layers[4].output)
    layer5_model_output = layer5_model.predict(input)

    layer6_model = tf.keras.Model(inputs = model.input, outputs = model.layers[5].output)
    layer6_model_output = layer6_model.predict(input)


    layer6_model_output_T = np.transpose(layer6_model_output)

After that, the weight for each layer was saved, and the weight for each layer was calculated through this.

This is a code that first outputs the calculated results in the form of a file.    

## AdamVCA

AdamVCA is an optimizer that combines Adaptive moment (Adam) and Vision Correction Algorithm (VCA).

Here are some things to focus on:

If you check AdamVCA with the current interpreter, you can see that a warning appears on line 236.

In the case of VCA, a metaheuristic optimization algorithm, the initial value is set first and then the search is performed.

An error occurs on line (236) when Epoch is not 0.

As mentioned above, VCA performs calculations after setting the initial value, so there was no problem when running the code in that part.

AdamVCA's code is uploaded, and the code consists of 5 hidden layers and 10 nodes per hidden layer. 
(The process of printing out comments and files is written in Korean. However, please keep in mind that this part does not put a strain on the code.)
