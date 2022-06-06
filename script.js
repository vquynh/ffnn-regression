/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    const N = 500;
    let data = Array(N);
    for (let i = 0; i < N; i++) {
        let randomSign = Math.random() < 0.5 ? -1 : 1;
        const randomNumber = Math.random();
        const x = randomSign*randomNumber;
        const noise = getRandomNoise(randomNumber, 0.03);
        const y = (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6) + noise;
        data[i] = {
            x: x,
            y: y
        }

    }
    return data;
}

function getRandomNoise(mean, variance) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    const number = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return variance*number;
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.x,
        y: d.y,
    }));

    tfvis.render.scatterplot(
        {name: 'y(x) = (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6)'},
        {values},
        {
            xLabel: 'x',
            yLabel: 'y(x)',
            height: 300
        }
    );

    // More code will be added below
    // Create the model
    if(model == null){
        model= createModel();
    }
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true, name: "input_layer"}));
    model.add(tf.layers.dense({units: 32, useBias: true, activation: "relu",  name: "layer_1"}));
    model.add(tf.layers.dense({units: 32, useBias: true, activation: "relu",  name: "layer_2"}));
    model.add(tf.layers.dense({units: 32, useBias: true, activation: "relu", name: "layer_3"}));
    model.add(tf.layers.dense({units: 32, useBias: true, activation: "relu", name: "layer_4"}));
    model.add(tf.layers.dense({units: 32, useBias: true, activation: "relu", name: "layer_5"}));

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

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse','accuracy'],
    });

    const batchSize = 32;
    const epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

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

    const originalPoints = inputData.map(d => ({
        x: d.x, y: d.y,
    }));


    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
            xLabel: 'x',
            yLabel: 'y(x)',
            height: 300
        }
    );
}

async function saveModel(){
    const result = await model.save('downloads://my-model');;
    console.log("result: ", result);
}
function getModelUrl(name){
    if(name === "Under Fitting"){
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/best-fitting.json"
    }
    if(name === "Best Fitting"){
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/best-fitting.json"
    }
    if(name === "Over Fitting"){
        return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/best-fitting.json"
    }
    return "https://raw.githubusercontent.com/vquynh/ffnn-regression/main/best-fitting.json"
}

async function loadModel(event){
    const modelName = event.target.value;
    const url = getModelUrl(modelName);
    model = await tf.loadLayersModel(url);
    console.log("Loaded model: ", modelName);
}
let model = createModel();
document.addEventListener('DOMContentLoaded', run);
document.getElementById("selectModel")
    .addEventListener('change', event => loadModel(event), false)

