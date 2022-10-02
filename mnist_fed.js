tf.setBackend("webgl")
const training_data_url = '/temporary_datasets/mnist_train.csv' 
const testing_data_url = '/temporary_datasets/mnist_test.csv' 
console.log(`The backend is ${tf.getBackend()}`)

function getModel(){
    const newModel = tf.sequential();
    this.datasetLength = 0
    newModel.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: [5, 5],
        activation: 'relu',
    }));
    newModel.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: [5, 5],
        activation: 'relu',
    }));
    newModel.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.25
    }));
    newModel.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: [3, 3],
        activation: 'relu',
    }));
    newModel.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: [3, 3],
        activation: 'relu',
    }));
    newModel.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.25
    }));
    newModel.add(tf.layers.flatten());

    newModel.add(tf.layers.dense({
        units: 256,
        activation: 'relu'
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.5
    }));
    newModel.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));

    const optimizer = 'rmsprop';
    newModel.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return newModel;
}

class FederatedModel{
    constructor(model_name){        
        this.name = model_name
        this.model = getModel();

        // tfvis.show.modelSummary({name:"Model Architecture"},this.model)

        this.training_history = []
        this.fitCallbacks = {}
        const {onEpochEnd, onBatchEnd} = tfvis.show.fitCallbacks({ name: 'Model Training' }, ['loss']);
        this.fitCallbacks["onTrainBegin"] = async (logs)=>{
            console.log("begun training [callback]")
        }
        this.fitCallbacks["onEpochEnd"] = async (epoch,logs)=>{
            console.log(`Epoch: ${epoch} loss: ${logs.loss}`)
            this.training_history.push(
                {
                    "Epoch":epoch,
                    "Loss":logs.loss
                }
            )
            console.log(this.training_history)
            onEpochEnd(epoch,logs)
        }
        // this.fitCallbacks["onBatchEnd"] = onBatchEnd 
        this.fitCallbacks["onBatchEnd"] = async (batch,logs)=>{
            console.log("processing batch.")
            onBatchEnd(batch,logs)
        }
        this.fitCallbacks["onEpochBegin"] = async (epoch,logs)=>{
            console.log(`Began ${epoch}`)
        }

    }
    //SIMULATED FEDERATED LEARNING SESSION FUNCTIONS
    async runSimulatedFL(C,K,r,epochs,clients){
        console.log("===STARTING SIMULATED FL SESSION===")
        console.log(`Number of Clients: ${K}`)
        console.log(`Sampling Rate: ${C}`)
        console.log(`Number of communication rounds: ${r}`)
        const nn = new FederatedModel() //main server model
        const nns = [] //will contain all the client models
        for(let i=0;i<K;i++){
            const modelObject = new FederatedModel(clients[i])
            nns.push(modelObject);
        }
        
        for(let t=0;t<r;t++){
            //for each communication round.
            console.log(`Communication round: ${t+1}`)
            let m = Math.floor(Math.max(...[C*K,1]))
            let index = _.sampleSize(_.range(K),m)
            console.log(`Chose clients: ${index}`)
            index.forEach((i,ind)=>{
                //dispatch server model weights to selected clients
                console.log(`Setting client: ${i} 's model to server model weights.`)
                nns[i].model.setWeights(nn.model.getWeights())    
            })
            //client_local_udpate
            
            for(let i=0;i<index.length;i++){
                const client_index = index[i]
                console.log(`Local update on client: ${client_index}`)
                const client_training_data_url = `/mnist-federated-dataset/client-${client_index}-train.csv`
                console.log(`Using dataset: ${client_training_data_url}`)
                nns[client_index] = await trainMnist(nns[client_index],epochs,client_training_data_url,false)
                const datasetLength = await getFederatedDatasetLength(client_training_data_url)
                nns[client_index].datasetLength = datasetLength
                console.log(`printing ${client_index} model`)
                console.log(nns[client_index])
                console.log(`testing model ${client_index}`)
                //testing local model with global test data
                await testMnist(null,nns[client_index])
            }
            //aggregation
            let s = 0;
            let aggregatedWeights = []
            for(let i=0;i<index.length;i++){
                const client_index = index[i]
                s+=nns[client_index].datasetLength
                if(i==(index.length-1)){
                    //last iteration
                    const weights = nns[client_index].model.getWeights()
                    for(let j=0;j<weights.length;j++){
                        aggregatedWeights.push(tf.zerosLike(weights[j]))
                    }
                }
            }
            for(let i=0;i<index.length;i++){
                const client_index = index[i]
                const weights = nns[client_index].model.getWeights()
                const ratio = nns[client_index].datasetLength/s
                console.log(`Aggregating client: ${client_index}`)
                for(let j=0;j<aggregatedWeights.length;j++){
                    const adjusted_weight = weights[j].mul(ratio)
                    aggregatedWeights[j] = aggregatedWeights[j].add(adjusted_weight)
                }   
            }
            nn.model.setWeights(aggregatedWeights)
        }
        console.log("testing global model")
        testMnist(null,nn)
    }

}
async function getFederatedDatasetLength(dataset_link){
    const dataset = await dfd.readCSV(dataset_link)
    const l = dataset.shape[0]
    return l-1
}
async function testMnist(modelName,modelObject){
    // const model = modelObject?modelObject.model:await tf.loadLayersModel(`localstorage://${modelName}`)
    let model = null
    if(modelObject){
        model = modelObject.model
    }else{
        model = await tf.loadLayersModel(`localstorage://${modelName}`) 
    }
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

async function trainMnist(modelObject,epochs,training_data_url,saveModel){
    //add param for dataset link
    console.log(`Training using ${training_data_url}`)
    const trainData = await tf.data.csv(
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
        console.log("TRAINING DATA EXAMPLE:")
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
    modelObject.training_history = []
    await modelObject.model.fitDataset((processedData), {
        epochs: epochs,
        // regist tensorboard for visualization
        callbacks:[
            modelObject.fitCallbacks,
        ]
    }).then(async () => {
        // save the current model
        if(saveModel){    
            const date = new Date()
            const modelName = `mnist-model-${date.getTime()}`
            await modelObject.model.save(`localstorage://${modelName}`)
            console.log(`Saved model as ${modelName} in localStorage`)
        }
    })
    if(!saveModel){
        return modelObject
    }
    console.log("Completed MNIST Training loop")
}

async function simulatedFederatedMain(){
    console.log("Starting federated main.")
    const C = 0.5; //sampling rate
    const K = 5; //number of clients
    const r = 1; //number of communication rounds
    const epochs_per_client = 10
    const clients = []
    for(let z=0;z<K;z++){
        clients.push(`client_model_${z}`)
    } 
    const fedModel = new FederatedModel()
    fedModel.runSimulatedFL(C,K,r,epochs_per_client,clients)
}

document.getElementById("train-mnist-normal").addEventListener("click",()=>{
    const fedModel = new FederatedModel("normal-model")
    tfvis.show.modelSummary({name:"Normal Training Loop Model Architecture"},fedModel.model)
    trainMnist(fedModel,1,training_data_url,true)
})
document.getElementById("test-mnist-normal").addEventListener("click",()=>{
    const modelName = prompt("Enter model's name.")
    const modelInLocalStorage = Object.keys(localStorage).filter(x=>x.includes("info")).some(x=>x.split("/")[1]==modelName)
    if(modelInLocalStorage){
        testMnist(modelName,null)
    }else{
        alert(`Model: ${modelName} not in local storage.`)
        console.error(`Model: ${modelName} not in local storage.`)
    }
})