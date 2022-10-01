tf.setBackend("webgl")
const training_data_url = '/temporary_datasets/mnist_train.csv' 
const testing_data_url = '/temporary_datasets/mnist_test.csv' 
console.log(`The backend is ${tf.getBackend()}`)

// const trainData = tf.data.csv(
//     training_data_url, {
//         hasHeader: true,
//         columnConfigs: {
//             label: {
//                 isLabel: true
//             }
//         }
//      } 
// );
// trainData.take(1).forEachAsync((d) => {
//     console.log(d)
// })
// const processedData = trainData.map(({
//     xs,
//     ys
// }) => {
//     return {
//         // get all pixels and put them into a tensor with 28 * 28 * 1
//         xs: tf.tensor(Object.values(xs), [28, 28, 1]).div(255),
//         // we need to do one-hot encoding for each label
//         ys: tf.oneHot((Object.values(ys)[0]), 10)
//     };
// }).shuffle(250).batch(64)
// // }).shuffle(1000).batch(64)

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

let training_history = []
const fitCallbacks = {}
const {onEpochEnd, onBatchEnd} = tfvis.show.fitCallbacks({ name: 'Model Training' }, ['loss']);
fitCallbacks["onTrainBegin"] = async (logs)=>{
    console.log("begun training [callback]")
}
fitCallbacks["onEpochEnd"] = async (epoch,logs)=>{
    console.log(`Epoch: ${epoch} loss: ${logs.loss}`)
    training_history.push(
        {
            "Epoch":epoch,
            "Loss":logs.loss
        }
    )
    console.log(training_history)
    onEpochEnd(epoch,logs)
}
fitCallbacks["onBatchEnd"] = onBatchEnd 
fitCallbacks["onEpochBegin"] = async (epoch,logs)=>{
    console.log(`Began ${epoch}`)
}

async function testMnist(modelName){
    const model = await tf.loadLayersModel(`localstorage://${modelName}`)
    const results = []
    const testData = await tf.data.csv(
            testing_data_url,{
                hasHeader:true,
                columnConfigs: {
                    label: {
                        isLabel: true
                    }
                }
            })
    const processedData = testData.map(({
        xs,
        ys
    }) => {
        return {
            // get all pixels and put them into a tensor with 28 * 28 * 1
            xs: tf.tensor(Object.values(xs), [28, 28, 1]).div(255),
            // we need to do one-hot encoding for each label
            ys: tf.oneHot((Object.values(ys)[0]), 10)
        };
    })
    const labelValues = []
    const predictedValues = []
    correctlyClassified = 0
    incorrectlyClassified = 0 
    await processedData.forEachAsync(({xs,ys})=>{
        const y = ys.arraySync()
        const labelValue = y.indexOf(Math.max(...y))
        // console.log(`Actual Label ${y.indexOf(Math.max(...y))}`)
        const prediction = model.predict(xs.reshape([1,28,28,1])).arraySync()[0]
        const predictionValue = prediction.indexOf(Math.max(...prediction))
        labelValues.push(labelValue)
        predictedValues.push(predictionValue)
        labelValue==predictionValue?correctlyClassified+=1:incorrectlyClassified+=1
        console.log("/10000 values processed")
    })
    labelValuesTensor = tf.tensor(labelValues)
    predictedValuesTensor = tf.tensor(predictedValues)
    console.log("Accuracy:")
    console.log(correctlyClassified/(correctlyClassified+incorrectlyClassified))
    const precision = tf.metrics.precision(labelValuesTensor,predictedValuesTensor)
    console.log("Precision:")
    console.log(precision.arraySync())
    const recall = tf.metrics.recall(labelValuesTensor,predictedValuesTensor)
    console.log("Recall:")
    console.log(recall.arraySync())
    console.log("done testing")
}

async function trainMnist(model,epochs){

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
    }).shuffle(250).batch(64)
    // }).shuffle(1000).batch(64)
    
    console.log("Train MNIST FUNCTION run.")
    training_history = []
    await model.fitDataset((processedData), {
        epochs: epochs,
        // regist tensorboard for visualization
        callbacks:[
            fitCallbacks,
        ]
    }).then(async () => {
        // save the current model
        const date = new Date()
        const modelName = `mnist-model-${date.getTime()}`
        await model.save(`localstorage://${modelName}`)
        console.log(`Saved model as ${modelName} in localStorage`)
    })
    console.log("Completed MNIST Training loop")
}

document.getElementById("train-mnist-normal").addEventListener("click",()=>{
    console.log("clicked!")
    trainMnist(model,1)
})