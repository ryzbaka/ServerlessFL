// const tf = require("@tensorflow/tfjs")

tf.setBackend("webgl")
const training_data_url = '/temporary_datasets/mnist_train.csv' 

const trainData = tf.data.csv(
    training_data_url, {
        hasHeader: true,
        columnConfigs: {
            label: {
                isLabel: true
            }
        }
     } 
);
trainData.take(1).forEachAsync((d) => {
    console.log(d)
})
const processedData = trainData.map(({
    xs,
    ys
}) => {
    return {
        // get all pixels and put them into a tensor with 28 * 28 * 1
        xs: tf.tensor(Object.values(xs), [28, 28, 1]).div(255),
        // we need to do one-hot encoding for each label
        ys: tf.oneHot((Object.values(ys)[0]), 10)
    };
}).shuffle(1000).batch(64)

const model = tf.sequential();
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: [5, 5],
    activation: 'relu',
}));
model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: [5, 5],
    activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
}));
model.add(tf.layers.dropout({
    rate: 0.25
}));
model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: [3, 3],
    activation: 'relu',
}));
model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: [3, 3],
    activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
}));
model.add(tf.layers.dropout({
    rate: 0.25
}));
model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    units: 256,
    activation: 'relu'
}));
model.add(tf.layers.dropout({
    rate: 0.5
}));
model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax'
}));

const optimizer = 'rmsprop';
model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

tfvis.show.modelSummary({name:"Model Architecture"},model)

const fitCallbacks = tfvis.show.fitCallbacks({ name: 'Model Training' }, ['loss']);

model.fitDataset((processedData), {
    epochs: 5,
    // regist tensorboard for visualization
    callbacks:[
        fitCallbacks,
        {
            //add stuff here to signal the beginning and ending of training loop
            onTrainBegin: async (logs)=>{
                console.log("training has begun")
            },
            onTrainEnd: async (logs)=>{
                console.log("training has ended")
            },
            onEpochEnd: async (epoch,logs)=>{
                console.log(`Epoch: ${epoch} Loss: ${logs.loss}`)
            }
        },
    ]
}).then(async () => {
    // save the current model
    const date = new Date()
    const modelName = `mnist-model-${date.getTime()}`
    await model.save(`localstorage://${modelName}`)
    console.log(`Saved model as ${modelName} in localStorage`)
})