
console.log("starting fedavg script...")

class BP{
    constructor(input_dim, batch_size,lr, file_name){
        this.input_dim = input_dim
        this.batch_size=batch_size //hard coded for now.
        this.filename = file_name //WHAT IS FILENAME USED FOR???
        // this.len = len //number of training data instances
        this.lr = lr //learning rate
        this.lossHistory = []
        // defining layers
        this.input = tf.zeros([batch_size, input_dim])
        this.w1 = tf.randomUniform([input_dim,20],-1,1)
        this.z1 = tf.randomUniform([batch_size,20])
        
        this.hidden_layer_1 = tf.zeros([batch_size,20])
        this.w2 = tf.randomUniform([20,20],-1,1)
        this.z2 = tf.randomUniform([batch_size,20],-1,1)
        
        this.hidden_layer_2 = tf.zeros([batch_size,20])
        this.w3 = tf.randomUniform([20,20],-1,1)
        this.z3 = tf.randomUniform([batch_size,20],-1,1)

        this.hidden_layer_3 = tf.zeros([batch_size,20])
        this.w4 = tf.randomUniform([20,1],-1,1)
        this.z4 = tf.randomUniform([batch_size,1],-1,1)
        
        this.output_layer = tf.zeros([batch_size,1])
        
        this.loss = tf.zeros([batch_size,1])
    }

    sigmoid(x){
        return tf.sigmoid(x)
    }

    sigmoid_derivative(x){
        //x*(1-x)
        return x.mul(-1).add(1).mul(x)
    }

    forward_prop(data,label){
        this.input = data
        this.z1 = this.input.dot(this.w1)
        
        this.hidden_layer_1 = this.sigmoid(this.z1)
        this.z2 = this.hidden_layer_1.dot(this.w2)
        
        this.hidden_layer_2 = this.sigmoid(this.z2)
        this.z3 = this.hidden_layer_2.dot(this.w3)
        
        this.hidden_layer_3 = this.sigmoid(this.z3)
        this.z4 = this.hidden_layer_3.dot(this.w4)

        this.output_layer = tf.tensor(this.sigmoid(this.z4).arraySync())//compute foward prop and store output as tensor
        this.loss = tf.tensor(this.output_layer.mul(-1).add(label).pow(2).mul(0.5).arraySync()) //compute huber loss and store as tensor
        return this.output_layer
    }
    
    getWeights(){
        const layers = { 
         "input":this.input,
         "w1":this.w1, 
         "z1":this.z1,
         "hidden_layer_1":this.hidden_layer_1,
         "w2":this.w2, 
         "z2":this.z2,
         "hidden_layer_2":this.hidden_layer_2,
         "w3":this.w3, 
         "z3":this.z3,
         "hidden_layer_3":this.hidden_layer_3,
         "w4":this.w4, 
         "z4":this.z4, 
         "output_layer":this.output_layer
        }
        Object.keys(layers).forEach(key=>{
            const intermediaryDfd = new dfd.DataFrame(tf.clone(layers[key]))
            const jsonTensor = dfd.toJSON(intermediaryDfd)
            layers[key] = jsonTensor
        })
        return layers
    }
    loadWeights(layers){
        Object.keys(layers).forEach(key=>{
            this[key] = tf.clone((new dfd.DataFrame(layers[key])).tensor)
        })   
    }
    clone(){
        const clonedNeuralNetwork = new BP(this.input_dim,this.batch_size,this.lr,this.file_name)
        clonedNeuralNetwork.loadWeights(this.getWeights())
        return clonedNeuralNetwork
    }
    backward_prop(label){
        //calculating derivate for w4
        const l_deri_out = this.output_layer.sub(label)
        const l_deri_z4 = l_deri_out.mul(this.sigmoid_derivative(this.output_layer))
        const l_deri_w4 = this.hidden_layer_3.transpose().dot(l_deri_z4)
        //calculating derivate for w3
        const l_deri_h3 = l_deri_z4.dot(this.w4.transpose())
        const l_deri_z3 = l_deri_h3.mul(this.sigmoid_derivative(this.hidden_layer_3))
        const l_deri_w3 = this.hidden_layer_2.transpose().dot(l_deri_z3)
        //calculating derivate for w2
        const l_deri_h2 = l_deri_z3.dot(this.w3.transpose())
        const l_deri_z2 = l_deri_h2.mul(this.sigmoid_derivative(this.hidden_layer_2))
        const l_deri_w2 = this.hidden_layer_1.transpose().dot(l_deri_z2)
        //calculating derivate for w1
        const l_deri_h1 = l_deri_z2.dot(this.w2.transpose())
        const l_deri_z1 = l_deri_h1.mul(this.sigmoid_derivative(this.hidden_layer_1))
        const l_deri_w1 = this.input.transpose().dot(l_deri_z1)

        //calculate updated weights and store as tensors
        this.w4 = tf.tensor(this.w4.sub((l_deri_w4.mul(this.lr))).arraySync())
        this.w3 = tf.tensor(this.w3.sub((l_deri_w3.mul(this.lr))).arraySync())
        this.w2 = tf.tensor(this.w2.sub((l_deri_w2.mul(this.lr))).arraySync())
        this.w1 = tf.tensor(this.w1.sub((l_deri_w1.mul(this.lr))).arraySync())
    }
}

async function loadDataset(){

    console.log("reading training dataset")
    let trainingData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/training_data.csv")//.then(data=>{
    let training_X = trainingData.iloc({columns:[`0:${trainingData.shape[1]-1}`],rows:[`0:${trainingData.shape[0]-1}`]})
    let training_y = trainingData.iloc({columns:[`${trainingData.shape[1]-1}:`],rows:[`0:${trainingData.shape[0]-1}`]})
    
    let validationData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/validation_data.csv")//.then(data=>{
    let validation_X = validationData.iloc({columns:[`0:${validationData.shape[1]-1}`],rows:[`0:${validationData.shape[0]-1}`]})
    let validation_y = validationData.iloc({columns:[`${validationData.shape[1]-1}:`],rows:[`0:${validationData.shape[0]-1}`]})
    
    let testingData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/testing_data.csv")//.then(data=>{
    let testing_X = testingData.iloc({columns:[`0:${testingData.shape[1]-1}`],rows:[`0:${testingData.shape[0]-1}`]})
    let testing_y = testingData.iloc({columns:[`${testingData.shape[1]-1}:`],rows:[`0:${testingData.shape[0]-1}`]})
    
    return [[training_X,training_y],[validation_X,validation_y],[testing_X,testing_y]]
}

function train(nn,epochs,training_X,training_y,testing_X,testing_y){
    nn.lossHistory = []
    nn.validationLossHistory
    nn.epochs = epochs
    batch = training_X.shape[0]/nn.batch_size
    //training
    min_epochs = 10
    best_model = null
    min_val_loss = 5
    for(let epoch=0;epoch<epochs;epoch++){
         let train_loss = []
        for(let i=0;i<batch;i++){
            let start = i*nn.batch_size
            let end = start+nn.batch_size
            // console.log(`Start: ${start}`)
            // console.log(`End: ${end}`)
            nn.forward_prop(training_X.iloc({rows:[`${start}:${end}`]}).tensor,training_y.iloc({rows:[`${start}:${end}`]}).tensor)
            nn.backward_prop(training_y.iloc({rows:[`${start}:${end}`]}).tensor)
        }
        train_loss.push(tf.mean(nn.loss).arraySync())
        nn.lossHistory.push(tf.mean(tf.tensor(train_loss)).arraySync())
        console.log(`Epoch: ${epoch+1} training_loss: ${tf.mean(tf.tensor(train_loss)).arraySync()}`)
    }
    //TO DO: Add validation for best model selection.
    // let trainedModel = nn;
    plotTrainingLoss(nn)
    test(nn,testing_X,testing_y)
    return nn
}

function plotTrainingLoss(nn){
    lossHistory = new dfd.DataFrame({"loss":nn.lossHistory})
    lossHistory.plot("plot_div").line()
}

function test(nn, testing_X,testing_y){
    //removed global variables
    if(!testing_X){
        console.error("No testing data.")
    }else{
        // let test_x = testing_X;
        // let test_y = testing_y
        let pred = []
        let batch = testing_y.shape[0]/nn.batch_size

        for(let i=0;i<batch;i++){
            let start = i * nn.batch_size
            let end = start+nn.batch_size
            let res = nn.forward_prop(testing_X.iloc({rows:[`${start}:${end}`]}).tensor,testing_y.iloc({rows:[`${start}:${end}`]}).tensor)
            pred = pred.concat(res.arraySync())
        }
        let predictionResult = tf.tensor(pred)
        // let predictionResultTensor = predictionResult
        //calculating MAE
        let mae = tf.sum(tf.abs(testing_y.tensor.sub(predictionResult))).arraySync()/predictionResult.shape[0]
        console.log(`MAE: ${mae}`)
        const comparisonData = new dfd.DataFrame({
            "actual": Array.from(testing_y.tensor.dataSync()),
            "predicted": Array.from(predictionResult.dataSync()),
        })
        comparisonData.plot("plot_test").line()
        return mae
    }
}

function saveModel(nn,name){
    //saving model to localStorage
    const modelObject = {
        weights: nn.getWeights(),
        input_dim: nn.input_dim,
        batch_size: nn.batch_size,
        lr: nn.lr,
        file_name: nn.file_name        
    }
    window.localStorage.setItem(name,JSON.stringify(modelObject))
}

function loadModel(name){
    const modelObject = JSON.parse(window.localStorage.getItem(name))
    const neuralNetwork = new BP(modelObject.input_dim, modelObject.batch_size, modelObject.lr,modelObject.file_name)
    neuralNetwork.loadWeights(modelObject.weights)
    return neuralNetwork
}

async function normalMain(){
    let [[training_X,training_y],[validation_X,validation_y],[testing_X,testing_y]] = await loadDataset()
    let neuralNetwork = new BP(input_dim=training_X.shape[1], batch_size=32,lr=0.01, file_name="test")
    let trainedNeuralNetwork = train(neuralNetwork,20,training_X,training_y,testing_X,testing_y)
    let modelName = (Math.random() + 1).toString(36).substring(7)
    saveModel(trainedNeuralNetwork,modelName)
    console.log(`Saved model as: ${modelName}`)
    testX = testing_X
    testY = testing_y
    // normalTrainedWeights =  trainedNeuralNetwork.getWeights() //normalTrainedWeights is a global variable that has to be deleted.

}
// Federated Learning implementation
async function loadFederatedDataset(index){
    console.log("reading training dataset")
    let trainingData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/training_data.csv")//.then(data=>{
    let training_X = trainingData.iloc({columns:[`0:${trainingData.shape[1]-1}`],rows:[`0:${trainingData.shape[0]-1}`]})
    let training_y = trainingData.iloc({columns:[`${trainingData.shape[1]-1}:`],rows:[`0:${trainingData.shape[0]-1}`]})
    
    let validationData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/validation_data.csv")//.then(data=>{
    let validation_X = validationData.iloc({columns:[`0:${validationData.shape[1]-1}`],rows:[`0:${validationData.shape[0]-1}`]})
    let validation_y = validationData.iloc({columns:[`${validationData.shape[1]-1}:`],rows:[`0:${validationData.shape[0]-1}`]})
    
    let testingData = await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/testing_data.csv")//.then(data=>{
    let testing_X = testingData.iloc({columns:[`0:${testingData.shape[1]-1}`],rows:[`0:${testingData.shape[0]-1}`]})
    let testing_y = testingData.iloc({columns:[`${testingData.shape[1]-1}:`],rows:[`0:${testingData.shape[0]-1}`]})
    
    return [[training_X,training_y],[validation_X,validation_y],[testing_X,testing_y]] 
}
class FedAvg{
    constructor(C,K,r,clients,input_dim,batch_size,lr){
        this.C = C
        this.K = K
        this.r = r
        this.clients = clients
        this.nn = new BP(input_dim, batch_size, lr, "server")
        this.nns = []

        for(let i=0;i<this.K;i++){
            //for each client create a copy of the neural networks
            console.log("Function is still being implemented.")
        }
    }
}

async function federatedMain(){
    const C = 0.5 // sampling rate
    const K = 10 //number of clients
    const r = 5 //number of communication rounds
    const clients = []
}
let testX = null
let testY = null
const normalTrain = document.querySelector("#normal_train").addEventListener("click",normalMain)
const federatedTrain = document.querySelector("#federated_train").addEventListener("click",federatedMain)