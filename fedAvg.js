
// const tf = require("@tensorflow/tfjs")
console.log("starting fedavg script...")

class BP{
    constructor(len,input_dim, batch_size,lr, file_name){
        this.batch_size=batch_size //hard coded for now.
        this.filename = file_name //WHAT IS FILENAME USED FOR???
        this.len = len //number of training data instances
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
        // console.log("===BATCH SHAPES===")
        // console.log(`X: ${data.shape}`)
        // console.log(`y: ${label.shape}`)
        // console.log("==================")

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
    
    getLayers(){
        const layers = [
            this.input,
            this.w1, 
            this.z1,
            this.hidden_layer_1,
            this.w2, 
            this.z2,
            this.hidden_layer_2,
            this.w3, 
            this.z3,
            this.hidden_layer_3,
            this.w4, 
            this.z4, 
            this.output_layer
        ]
        layers.forEach((layer,index)=>{
            console.log("==========")
            console.log(`Layer ${index}:`)
            console.log(layer)
            console.log("==========")
        })
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

function load_data(){
    //load a csv data file
    console.error("this function has not been implemented yet") 
}
async function loadDataset(){

    console.log("reading training dataset")
    await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/training_data.csv").then(data=>{
        training_X = data.iloc({columns:[`0:${data.shape[1]-1}`],rows:[`0:${data.shape[0]-1}`]})
        training_y = data.iloc({columns:[`${data.shape[1]-1}:`],rows:[`0:${data.shape[0]-1}`]})
        console.log("done reading training dataset")
    })
    await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/validation_data.csv").then(data=>{
        validation_X = data.iloc({columns:[`0:${data.shape[1]-1}`],rows:[`0:${data.shape[0]-1}`]})
        validation_y = data.iloc({columns:[`${data.shape[1]-1}:`],rows:[`0:${data.shape[0]-1}`]})
        console.log("done reading validation dataset")          
    })
    await dfd.readCSV("https://raw.githubusercontent.com/ryzbaka/ServerlessFL/main/testing_data.csv").then(data=>{
        testing_X = data.iloc({columns:[`0:${data.shape[1]-1}`],rows:[`0:${data.shape[0]-1}`]})
        testing_y = data.iloc({columns:[`${data.shape[1]-1}:`],rows:[`0:${data.shape[0]-1}`]})
        console.log("done reading testing dataset")  
    })
    return "loaded datasets"
}

function validateDataLoad(){
    if (training_X && training_y && validation_X && validation_y && testing_X && testing_y){
        return true
    }else{
        return false
    }
}


function train(nn,epochs){
    if(!validateDataLoad()){
        console.error("No datasets loaded!!!")
        return null
    }
    console.log("TRAINING STARTED")    
    console.log(`Training X shape ${training_X.tensor.shape}`)
    console.log(`Training y shape ${training_X.tensor.shape}`)

    console.log(`Testing X shape ${testing_X.tensor.shape}`)
    console.log(`Testing y shape ${testing_y.tensor.shape}`)

    console.log(`Validation X shape ${validation_X.tensor.shape}`)
    console.log(`Validation y shape ${validation_y.tensor.shape}`)
    nn.lossHistory = []
    nn.validationLossHistory
    nn.len = training_X.shape[0]
    // nn.batch_size = batch_size
    nn.epochs = epochs
    batch = nn.len/nn.batch_size
    //training
    min_epochs = 10
    best_model = null
    min_val_loss = 5
    for(let epoch=0;epoch<epochs;epoch++){
        train_loss = []
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
    trainedModel = nn;
    plotTrainingLoss(nn)
    test(trainedModel)
    return trainedModel
}

function plotTrainingLoss(nn){
    lossHistory = new dfd.DataFrame({"loss":nn.lossHistory})
    lossHistory.plot("plot_div").line()
}

function test(nn){
    if(!testing_X){
        console.error("No testing data.")
    }else{
        let test_x = testing_X;
        let test_y = testing_y
        let pred = []
        let batch = test_y.shape[0]/nn.batch_size

        for(let i=0;i<batch;i++){
            let start = i * nn.batch_size
            let end = start+nn.batch_size
            res = nn.forward_prop(testing_X.iloc({rows:[`${start}:${end}`]}).tensor,testing_y.iloc({rows:[`${start}:${end}`]}).tensor)
            pred = pred.concat(res.arraySync())
        }
        predictionResult = tf.tensor(pred)
        // let predictionResultTensor = predictionResult
        //calculating MAE
        let mae = tf.sum(tf.abs(testing_y.tensor.sub(predictionResult))).arraySync()/predictionResult.shape[0]
        console.log(`MAE: ${mae}`)
        const comparisonData = new dfd.DataFrame({
            "actual": Array.from(training_y.tensor.dataSync()),
            "predicted": Array.from(predictionResult.dataSync()),
        })
        comparisonData.plot("plot_test").line()
        return mae
    }
}

function normalMain(){
    loadDataset().then(data=>{
        console.log(data)
        const neuralNetwork = new BP(len=training_X.shape[0],input_dim=training_X.shape[1], batch_size=32,lr=0.01, file_name="test")
        train(neuralNetwork,10)
    })
}
// Federated Learning implementation
class FedAvg{
    constructor(sampling_rate,number_of_clients,clients){
        this.C = sampling_rate
        this.K = number_of_clients
        this.clients = clients
        this.nn = new BP()
        this.nns = []
        //distribution

    }
}

function loadFederatedDataset{
    console.error("Function not implemented yet.")
}

function federatedMain(){
    // let clients = []
    // for(let i=1;i<11;i++){
    //     clients.push(`Task1_W_Zone${i}`)
    // }    
    // let sampling_rate = 0.5
    // let number_of_clients = 10
    // fed = new FedAvg()    
    console.error("Function not implemented yet.")
}
//all these global variables have to be stored in leveldb at the end of a main function
let training_X = null
let training_y = null
let validation_X = null
let validation_y = null
let testing_X = null
let testing_y = null
let trainedModel = null
let testingResult = null
let predictionResult = null
const normalTrain = document.querySelector("#normal_train").addEventListener("click",normalMain)