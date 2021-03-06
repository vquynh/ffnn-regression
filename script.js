/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
function getData(samples, variance) {
    const N = samples;
    let data = Array(N);
    for (let i = 0; i < N; i++) {
        let randomSign = Math.random() < 0.5 ? -1 : 1;
        const randomNumber = Math.random();
        const x = randomSign*randomNumber;
        const noise = getRandomNoise(variance);
        const y = (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6) + noise;
        data[i] = {
            x: x,
            y: y
        }
    }
    return data;
}

function getRandomNoise(variance) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    const number = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return Math.sqrt(variance)*number;
}

function setDefaultParameters(){
    activation = "relu";
    document.getElementById("selectActivation").value = activation;
    trainingSamples = 100;
    document.getElementById("selectTrainingSample").value = trainingSamples;
    trainingVariance = 0.001;
    document.getElementById("selectTrainingVariance").value = trainingVariance;
    testingSamples = 100;
    document.getElementById("selectTestingSample").value = testingSamples;
    testingVariance = 0.001;
    document.getElementById("selectTestingVariance").value = testingVariance;
    epochs = 100;
    document.getElementById("selectEpochs").value = epochs;
    learningRate = 0.001;
    document.getElementById("selectLearningRate").value = learningRate;
}

function setParametersByModel(name) {
    if(name === "Under Fitting"){
        nLayers = 1;
        neurons = 2;
    }else if(name === "Over Fitting"){
        nLayers = 10;
        neurons = 256;
        trainingSamples = 50;
        trainingVariance = 0.01;
        testingSamples = 50;
        testingVariance = 0.001;
        learningRate = 0.003;
        document.getElementById("selectTrainingSample").value = trainingSamples;
        document.getElementById("selectTrainingVariance").value = trainingVariance;
        document.getElementById("selectTestingSample").value = testingSamples;
        document.getElementById("selectTestingVariance").value = testingVariance;
        document.getElementById("selectLearningRate").value = learningRate;
    }else{
        nLayers = 5;
        neurons = 32;
    }
    document.getElementById("selectHiddenLayers").value = nLayers;
    document.getElementById("selectNeurons").value = neurons;
}

async function run() {
    const selectedModel = document.getElementById("selectModel").value || "Best Fitting";
    setParametersByModel(selectedModel);
    setDefaultParameters();

    // Get pre-trained model
    model = await getPreTrainedModel(selectedModel);
    await showSummaryAndTestModel();
}

async function showSummaryAndTestModel(){
    tfvis.show.modelSummary(document.getElementById("summary"), model);
    console.log("Model with layers: ", nLayers);
    await testModel(model);
}

async function createTrainAndTestNewModel(){
    model = createModel();
    tfvis.show.modelSummary(document.getElementById("summary"), model);
    console.log("Model with layers: ", nLayers);

    await trainModel();
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true, name: "input_layer"}));

    // Add n hidden layers with given number of neurons and activation function
    for (let i = 0; i < nLayers; i++) {
        model.add(tf.layers.dense({
            units: neurons,
            useBias: true,
            activation: activation,
            name: "hidden_layer_" + (i + 1)
        }));
    }

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true, name: "output_layer"}));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

function plotLossGraph() {
    return tfvis.show.fitCallbacks(
        {drawArea: document.getElementById("trainingPerformance")},
        ['loss'],
        {height: 200, callbacks: ['onEpochEnd']}
    );
}

async function trainModel() {
    // Load and plot the original input data that we are going to train on.
    const trainingData = getData(trainingSamples, trainingVariance);

    tfvis.render.scatterplot(
        {drawArea: document.getElementById("trainingInput")},
        {values: trainingData, series:["Training data"]},
        {
            xLabel: 'x',
            yLabel: 'y(x)',
            height: 300
        }
    );

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(trainingData);
    const {inputs, labels} = tensorData;

    await model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse','accuracy'],
    });

    const batchSize = 32;

    // Train model
    await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: plotLossGraph()
    });

    await testModel();
}

async function testModel() {
    const testingData = getData(testingSamples, testingVariance);
    const tensorData = convertToTensor(testingData);
    const {inputMax, inputMin, labelMin, labelMax} = tensorData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    const originalPoints = testingData.map(d => ({
        x: d.x, y: d.y,
    }));

    tfvis.render.scatterplot(
        {drawArea: document.getElementById("result")},
        {values: [originalPoints, predictedPoints], series: ['Test data', 'predicted']},
        {
            xLabel: 'x',
            yLabel: 'y(x)',
            height: 300
        }
    );
}

async function saveModel(){
    await model.save('downloads://my-model');
}
function getModelUrl(name){
    if(name === "Under Fitting"){
        console.log("Got: ", name);
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/under-fitting.json";
    }
    if(name === "Best Fitting"){
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/best-fitting.json";
    }
    if(name === "Over Fitting"){
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/over-fitting.json";
    }
}

async function getPreTrainedModel(modelName){
    const url = getModelUrl(modelName);
    return await tf.loadLayersModel(url);
}

async function selectModel(modelName){
    setParametersByModel(modelName);
    model = await getPreTrainedModel(modelName);
    await showSummaryAndTestModel();
}

async function changeTestingSamples(value) {
    testingSamples = Number(value);
    await testModel();
}
async function changeTestingVariance(value) {
    testingVariance = Number(value);
    await testModel();
}

async function changeTrainingSamples(value) {
    trainingSamples = Number(value);
    await trainModel();
}
async function changeTrainingVariance(value) {
    trainingVariance = Number(value);
    await trainModel();
}

async function changeActivation(value) {
    activation = value.toLowerCase();
    await createTrainAndTestNewModel();
}
async function changeLayers(value) {
    nLayers = Number(value);
    await createTrainAndTestNewModel();
}
async function changeNeurons(value) {
    neurons = Number(value);
    await createTrainAndTestNewModel();
}
async function changeEpochs(value) {
    epochs = Number(value);
    await trainModel();
}
async function changeLearningRate(value) {
    learningRate = Number(value);
    await trainModel();
}
let model, trainingVariance, trainingSamples, testingVariance, testingSamples, activation, nLayers, neurons, epochs, learningRate;
document.addEventListener('DOMContentLoaded', run);
document.getElementById("selectModel")
    .addEventListener('change', event => selectModel(event.target.value), false);
// this triggers new testing
document.getElementById("selectTestingSample")
    .addEventListener('change', event => changeTestingSamples(event.target.value), false);
// this triggers new testing
document.getElementById("selectTestingVariance")
    .addEventListener('change', event => changeTestingVariance(event.target.value), false);
// this triggers new training and testing
document.getElementById("selectTrainingSample")
    .addEventListener('change', event => changeTrainingSamples(event.target.value), false);
// this triggers new testing
document.getElementById("selectTrainingVariance")
    .addEventListener('change', event => changeTrainingVariance(event.target.value), false);
// this change creates new model
document.getElementById("selectActivation")
    .addEventListener('change', event => changeActivation(event.target.value), false);
// this change creates new model
document.getElementById("selectHiddenLayers")
    .addEventListener('change', event => changeLayers(event.target.value), false);
// this change creates new model
document.getElementById("selectNeurons")
    .addEventListener('change', event => changeNeurons(event.target.value), false);
// this triggers new training and testing
document.getElementById("selectEpochs")
    .addEventListener('change', event => changeEpochs(event.target.value), false);
// this triggers new training and testing
document.getElementById("selectLearningRate")
    .addEventListener('change', event => changeLearningRate(event.target.value), false);

