tf.setBackend("webgl")
tf.enableProdMode()
localStorage.clear()

// const training_data_url = '/temporary_datasets/mnist_train.csv' 
const testing_data_url = '/temporary_datasets/mnist_test.csv' 
console.log(`The backend is ${tf.getBackend()}`)
const initializer = tf.initializers.glorotUniform({"seed":42})
function sleep(miliseconds) {
    var currentTime = new Date().getTime();
 
    while (currentTime + miliseconds >= new Date().getTime()) {
    }
}
function getModel(){
    const newModel = tf.sequential();
    this.datasetLength = 0
    newModel.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: [5, 5],
        activation: 'relu',
        kernelInitializer:initializer,
        biasInitializer:initializer
    }));
    newModel.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: [5, 5],
        activation: 'relu',
        kernelInitializer:initializer,
        biasInitializer:initializer
    }));
    newModel.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.25,
        seed:42
    }));
    newModel.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: [3, 3],
        activation: 'relu',
        kernelInitializer:initializer,
        biasInitializer:initializer
    }));
    newModel.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: [3, 3],
        activation: 'relu',
        kernelInitializer:initializer,
        biasInitializer:initializer
    }));
    newModel.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.25,
        seed:42
    }));
    newModel.add(tf.layers.flatten());

    newModel.add(tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelInitializer:initializer,
        biasInitializer:initializer
    }));
    newModel.add(tf.layers.dropout({
        rate: 0.5,
        seed:42
    }));
    newModel.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer:initializer,
        biasInitializer:initializer
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
            showMessage("begun training [callback]")
            showMessage((new Date().toTimeString()))
        }
        this.fitCallbacks["onEpochEnd"] = async (epoch,logs)=>{
            showMessage(`Epoch: ${epoch} loss: ${logs.loss}`)
            showMessage((new Date().toTimeString()))
            this.training_history.push(
                {
                    "Epoch":epoch,
                    "Loss":logs.loss,
                }
            )
            // console.log(this.training_history)
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
                // const client_training_data_url = `/mnist_non_iid/client-${client_index}-train.csv`
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
    }).batch(64)
    const res = await model.evaluateDataset(processedData)
    showMessage("Accuracy:")
    showMessage(res[1].arraySync())
    // const labelValues = []
    // const predictedValues = []
    // correctlyClassified = 0
    // incorrectlyClassified = 0 
    // await processedData.forEachAsync(({xs,ys})=>{
    //     const y = ys.arraySync()
    //     const labelValue = y.indexOf(Math.max(...y))
    //     // console.log(`Actual Label ${y.indexOf(Math.max(...y))}`)
    //     const predictionTensor= model.predict(xs.reshape([1,28,28,1]))
    //     // const prediction = model.predict(xs.reshape([1,28,28,1])).arraySync()[0]
    //     const prediction = predictionTensor.arraySync()[0]
    //     predictionTensor.dispose()
    //     const predictionValue = prediction.indexOf(Math.max(...prediction))
    //     // labelValues.push(labelValue)
    //     // predictedValues.push(predictionValue)
    //     labelValue==predictionValue?correctlyClassified+=1:incorrectlyClassified+=1
    //     xs.dispose()
    //     ys.dispose()
    //     console.log("/10000 values processed")

    // })
    //--------x
    // labelValuesTensor = tf.tensor(labelValues)
    // predictedValuesTensor = tf.tensor(predictedValues)
    // showMessage("Accuracy:")
    // showMessage(correctlyClassified/(correctlyClassified+incorrectlyClassified))
    // const precision = tf.metrics.precision(labelValuesTensor,predictedValuesTensor)
    // console.log("Precision:")
    // console.log(precision.arraySync())
    // const recall = tf.metrics.recall(labelValuesTensor,predictedValuesTensor)
    // console.log("Recall:")
    // console.log(recall.arraySync())
    console.log("done testing")
}

async function trainMnist(modelObject,epochs,training_data_url,saveModel){
    //add param for dataset link
    console.log(`===Training using ${training_data_url}===`)
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
    }).shuffle(250,42,true).batch(64)
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
            // await modelObject.model.save(`localstorage://${modelName}`)
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
    showMessage(`Running simulated federated learning - sampling_rate:${C} ; number_of_simulated_clients:${K} ; epochs_per_client:${epochs_per_client}`)
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

class PeerNode{
    constructor(id){
        this.peer = new Peer(id)
        this.id = id
        this.db = new LocalDatabase(`${id}-db`)
        this.initFederatedLearning(`${id}-model`)
        this.peer.on("open",(id)=>{
            showMessage(`This Peer ${id} connected to brokering server.`)
            document.title = `${id}-node`
        })
        this.K = parseInt(prompt("Enter the number of clients you'd like to you for the federation:")) 
        // this.sampling_rate = 1
        this.sampling_rate = 0.5 
        this.selected_clients = []
        this.weights_queue = []
        this.encrypted_message_queue = []
        this.peer.on("disconnected",async ()=>{
            showError("Disconnected from brokering server")
            // this.peer = new Peer(this.peer.id)
            // await this.connectToPeers()
            // console.log("done reestablishing connections")
        })
        // this.locked = false
        this.addressBook = {}
        this.peer.on("connection",connection=>{
            //someone else initiated connection
            connection.on("open",()=>{
                showMessage(`*Connected to ${connection.peer}`)
            })
            connection.on("close",()=>{
                showError(`*Disconnected from ${connection.peer}`)
                //unreliable, dont add anything else here
            })
            connection.on("data",(data)=>{
                this.handleData(connection,data)
            })
        })
        //Error handling for unavailable peer
        this.peer.on("error",(err)=>{
            if(err.type=="peer-unavailable"){
                showError(err.message)
            }       
        })
    }
    async connectToPeers(){
        const peers = Object.keys(this.addressBook)
        for(let i=0;i<peers.length;i++){
            await this.connectTo(peers[i])
        }
    }
    async initiateFederatedSession(sampling_rate=0.5,num_rounds,epochs_per_client,noiseScale,resetModel){
        this.sampling_rate = sampling_rate
        this.rounds = num_rounds
        this.epochs_per_client = epochs_per_client
        this.noiseScale = noiseScale
        if(resetModel){
            this.initFederatedLearning(`${this.id}-model`)
            localStorage.clear()
        }
        // const noiseScale =
        this.weights_queue = []
        //distribute weights to peers in address book
        const peer_ids = Object.keys(this.addressBook)
        this.selected_clients = _.sampleSize(peer_ids,Math.floor(Math.max(...[this.sampling_rate*this.K,1])))
        console.log(`Selected clients:${this.selected_clients.toString()}`)
        // const epochs_per_client = 3
        const initiator_name = this.id
        // function sleep(miliseconds) {
        //     var currentTime = new Date().getTime();
         
        //     while (currentTime + miliseconds >= new Date().getTime()) {
        //     }
        // }
        if(peer_ids.length>=this.K){
            this.selected_clients.forEach(async (el,index)=>{
                // sleep(5000)
                await this.sendWeightsToPeer(el,epochs_per_client,initiator_name,noiseScale,this.rounds)
                sleep(5000)
            })
        }else{
            alert(`${this.K} not available`)
        }
    }
    async sendWeightsToPeer(id,epochs_per_client,initiator_name,noiseScale,roundsleft){
        const weights = this.model.getWeights()
        const arrayfied_weights = []
        weights.forEach((el,index)=>{
            arrayfied_weights.push(el.arraySync())
        })
        this.sendEncryptedMessage(id,JSON.stringify(
            {
                "message_type":"sent-weights",
                "message_content":{
                    "weights":arrayfied_weights,
                    "epochs":epochs_per_client,
                    "initiator_id":initiator_name,
                    "noiseScale":noiseScale,
                    "roundsleft":roundsleft
                }
            }
        ))
    }
    getTrainingHistoryCSV(){
        alert("not implemented")
    }
    initFederatedLearning(model_name){        
        this.model_name = model_name
        this.model = getModel();

        // tfvis.show.modelSummary({name:"Model Architecture"},this.model)

        this.training_history = []
        this.fitCallbacks = {}
        const {onEpochEnd, onBatchEnd} = tfvis.show.fitCallbacks({ name: `Model Training - ${this.model_name}` }, ['loss']);
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
    connectTo(id){
        const connection = this.peer.connect(id)
        //this peer initiated connection
        connection.on("open",()=>{
            showMessage(`Connected to ${connection.peer}`)
        })
        connection.on("close",()=>{
            showError(`Disconnected from ${connection.peer}`)
            //unreliable don't add anything else here
        })
        connection.on("data",data=>{
            this.handleData(connection,data)
        })
    }
    sendMessage(id,data){
        console.log(`sending message ${data} to ${id}`)
        this.peer.connections[id].slice(-1)[0].send(JSON.stringify(data)) 
    }
    sendEncryptedMessage(id,message){
        console.log(`sending encrypted message to ${id}`)
        const encryptedMessage = this.encryptMessageForPeer(id,JSON.stringify(message))
        if(encryptedMessage){
            this.sendMessage(id,{
                "message_type":"encrypted_message",
                "encrypted_message":encryptedMessage
            })
        }
    }
    arrayStringToArray(s){
        console.log("in arrayStringToArray")
        console.log(s)
        const splitString = s.split(",")
        const arr = []
        splitString.forEach(el=>{
            arr.push(parseInt(el))
        })
        return arr
    }
    handleData(connection,data){
        showMessage(`Received from ${connection.peer}: ${JSON.parse(data).message_type}`)
        // if(this.locked){
        if(false){
            showError(`Did not respond to message: ${data} from ${connection.peer} (${this.peer.id} is locked.)`)
        }else{
            // this.locked = true
            // showMessage(`Locked ${this.peer.id}`)
            const messageObject = JSON.parse(data)
            if(messageObject.message_type==="join-fed"){
                //check address book for peer with id already exists
                if(Object.keys(this.addressBook).includes(connection.peer)){
                    showError(`${connection.peer} already in federation`)
                }else{
                    //if not exist, generate prime, generator and keys 
                    showMessage(`initiating DH key share`)
                    // this.lock = true
                    this.addressBook[connection.peer] = {
                        "dhObject": cryptojs.createDiffieHellman(256), //delete after verify shared
                        "key-verified":false //set to true once shared key matched
                    }
                    // showMessage("Generating key, prime and generator")
                    const keys = this.addressBook[connection.peer]["dhObject"].generateKeys()
                    // showMessage("Generated key")
                    // console.log("my key is:")
                    console.log(keys)
                    const prime = this.addressBook[connection.peer]["dhObject"].getPrime().toJSON().data
                    // showMessage("Generated prime")
                    const generator = this.addressBook[connection.peer]["dhObject"].getGenerator().toJSON().data
                    // showMessage("Generated generator")
                    // showMessage(`Sending key, prime and generator to ${connection.peer}`)
                    this.sendMessage(connection.peer,{
                        "message_type":"join-fed-init-dh",
                        "keys":keys.toJSON().data,
                        "prime":prime,
                        "generator":generator
                    })
                }
            }else if(messageObject.message_type==="join-fed-init-dh"){
                //IMPLEMENTING
                // const otherKeys = this.arrayStringToArray(messageObject.keys)
                const otherKeys = messageObject.keys
                // const prime = this.arrayStringToArray(messageObject.prime)
                const prime = messageObject.prime
                // const generator = this.arrayStringToArray(messageObject.generator)
                const generator = messageObject.generator
                this.addressBook[connection.peer] = {
                    "dhObject":cryptojs.createDiffieHellman(prime,generator)
                }
                // showMessage(`${this.id} is generating its keys`)
                
                const myKeys = this.addressBook[connection.peer]["dhObject"].generateKeys()
                
                // showMessage(`${this.id} generated it's keys`)
                // showMessage(`${this.id} is computing secret`)
                
                const mySecret = this.addressBook[connection.peer]["dhObject"].computeSecret(otherKeys)
                
                // showMessage(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                // console.log(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                this.addressBook[connection.peer] = {
                    "secret":mySecret.toString()
                }
                this.db.putObject("addressBook",this.addressBook)
                this.sendMessage(connection.peer,{
                    "message_type":"join-fed-final-dh",
                    "keys":myKeys.toJSON().data
                })
                
                // showMessage(`${this.id} has computed its secret`)
            }else if(messageObject.message_type==="join-fed-final-dh"){
                // const otherKeys = this.arrayStringToArray(messageObject.keys)
                const otherKeys = messageObject.keys
                // showMessage(`${this.id} is computing its secret`)
                const mySecret = this.addressBook[connection.peer]["dhObject"].computeSecret(otherKeys)
                // showMessage(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                // console.log(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                this.addressBook[connection.peer] = {
                    "secret":mySecret.toString()
                }
                this.db.putObject("addressBook",this.addressBook)
                // showMessage(`${this.id} has computed its secret`)
            }
            else if(messageObject.message_type==="ping"){
                this.sendMessage(connection.peer,{
                    "message-type":"ping",
                })
            }
            else if(messageObject.message_type=="encrypted_message"){
                // if(!this.locked){ -->do not uncomment this
                    // showMessage(`${this.peer.id} locked}`)
                    // showMessage(`Processing encrypted message from ${connection.peer}`)
                    const decryptedMessage = this.decryptMessageFromPeer(connection.peer, messageObject.encrypted_message)
                    // const decryptedMessageObject = JSON.parse(decryptedMessage)
                    // console.log(decryptedMessageObject) 
                    const decryptedMessageObject = JSON.parse(decryptedMessage)
                    console.log(decryptedMessageObject)
                    this.processEncryptedMessage(connection,decryptedMessageObject)
                // }else{
                    // showMessage("Received encrypted message while locked, storing in message_queue")
                    // this.encrypted_message_queue.push({
                    //     "connection":connection,
                    //     "encrypted_message":messageObject.encrypted_message
                    // })
                // }
            }
            // this.locked = false
            // showMessage(`${this.peer.id}'s lock has been unlocked`)
            // this.processEncryptedMessageQueue()
            // console.log(`${this.peer.id}'s  has been unlocked`)
        }
    }
    async processEncryptedMessageQueue(){
        while(this.encrypted_message_queue.length>0){
            console.log("processing encrypted message queue")
            //dequeue
            const queue_object = this.encrypted_message_queue[0]
            this.encrypted_message_queue = this.encrypted_message_queue.slice(1,)
            // processing dequeued encrypted message
            // this.lock = true
            // showMessage(`Locked ${this.peer.id}`)
            // showMessage(`Processing encrypted message from ${queue_object.connection.peer}`)
            const decryptedMessage = this.decryptMessageFromPeer(queue_object.connection.peer, queue_object.encrypted_message)
            // const decryptedMessageObject = JSON.parse(decryptedMessage)
            // console.log(decryptedMessageObject) 
            const decryptedMessageObject = JSON.parse(decryptedMessage)
            this.processEncryptedMessage(queue_object.connection,decryptedMessageObject)
            console.log(decryptedMessageObject)
            this.processEncryptedMessage(queue_object.connection,decryptedMessageObject)       
            // this.lock = false
            // showMessage(`${this.peer.id}'s lock has been unlocked`)
        }
    }
    async processEncryptedMessage(connection,message){
        showMessage("processing encrypted message")
        const messageObject = JSON.parse(message)
        console.log(messageObject)
        console.log(`Message type is ${messageObject["message_type"]}`)
        if(messageObject["message_type"]=="text-message"){
            console.log(`processing text message`)
            // showMessage(`${connection.peer}: ${messageObject.message_content}`)
        }else if(messageObject["message_type"]=="sent-weights"){
            showMessage("Received weights")
            const initiator_id = messageObject.message_content.initiator_id
            const arrayfied_weights = messageObject.message_content.weights 
            const roundsleft = messageObject.message_content.roundsleft
            const tensored_weights = []
            arrayfied_weights.forEach((element,index)=>{
                tensored_weights.push(tf.tensor(element))
            })
            this.model.setWeights(tensored_weights)
            showMessage("received global weights")
            this.runLocalUpdate(messageObject.message_content.epochs,initiator_id,messageObject.message_content.noiseScale,roundsleft)
        }else if(messageObject["message_type"]=="sent-weights-for-aggregation"){
            const peer_id = messageObject.message_content.peer_id
            const roundsleft = messageObject.message_content.roundsleft
            console.log(`Received weights for aggregation from ${peer_id}`)
            console.log(messageObject.message_content)
            const datasetLength = messageObject.message_content.datasetLength
            const arrayfied_weights = messageObject.message_content.weights            
            const tensored_weights = []
            arrayfied_weights.forEach((element,index)=>{
                tensored_weights.push(tf.tensor(element))
            })
            if(Object.keys(this.weights_queue).map(v=>parseInt(v)).includes(roundsleft)){
                this.weights_queue[roundsleft].push(
                    {
                        "peer_id":peer_id,
                        "datasetLength":datasetLength,
                        "weights":tensored_weights,
                    }
                )
            }else{
                this.weights_queue[roundsleft] = []
                this.weights_queue[roundsleft].push(
                    {
                        "peer_id":peer_id,
                        "datasetLength":datasetLength,
                        "weights":tensored_weights,
                    }
                )
            }
            if(this.weights_queue[roundsleft].length==this.selected_clients.length){
                this.aggregateWeights(roundsleft)
            }
            // this.weights_queue.push(
            //     {
            //         "peer_id":peer_id,
            //         "datasetLength":datasetLength,
            //         "weights":tensored_weights,
            //         "roundsleft":roundsleft      
            //     }
            // )
            // console.log("weights queue:")
            // console.log(this.weights_queue)
            
            // this.weights_queue = [...new Map(node.weights_queue.map(item =>
            //     [item["peer_id"], item])).values()];
            // if(this.weights_queue.length==(this.selected_clients.length))
            // {
            //     this.aggregateWeights()
            // }
        }else if(messageObject["message_type"]=="sent-aggregated-weights"){
            showMessage(`Received aggregated weights from ${messageObject.message_content.sender}`)
            const receivedAggregatedWeights = messageObject.message_content.weights
            showMessage("setting weights of local model")
            const aggWeights = []
            receivedAggregatedWeights.forEach((el,index)=>{
                aggWeights.push(tf.tensor(el))
            })
            this.model.setWeights(aggWeights)
            showMessage(`set local model weights to aggregated weights received from ${messageObject.message_content.sender}`)
        }
    }
    async aggregateWeights(roundsleft){
        //aggregating
        showMessage("Weights queue length satisfied, aggregating weights")
        let s = 0;
        let datasetLengths = []
        let trained_weights = []
        let aggregatedWeights = [] //zero array of same shape as that of the received weights
        //populate aggregatedWeights with zeros
        // showMessage("populating aggregatedWeights with zeros")
        // const weights = this.weights_queue[0].weights
        const weights = this.weights_queue[roundsleft][0].weights //seleting the first weights array just to create a zeros array like it.
        for(let i=0;i<weights.length;i++){
            aggregatedWeights.push(tf.zerosLike(weights[i]))
        }
        //calculate total number of instances used to train the weights
        // showMessage("calculating total number of instances used to train the weights")
        for(let i=0;i<this.weights_queue[roundsleft].length;i++){
            s+=this.weights_queue[roundsleft][i].datasetLength
        }
        // aggregate weights
        showMessage("aggregating weights")
        for(let i=0;i<this.weights_queue[roundsleft].length;i++){
            const model_weights = this.weights_queue[roundsleft][i].weights
            const ratio = this.weights_queue[roundsleft][i].datasetLength/s
            showMessage(`Aggregating weights from: ${this.weights_queue[roundsleft][i].peer_id}`)
            for(let j=0;j<model_weights.length;j++){
                const adjusted_weights = model_weights[j].mul(ratio)
                aggregatedWeights[j] = aggregatedWeights[j].add(adjusted_weights)    
            }
        }
        showMessage(`Setting ${this.id}'s weights to the aggregated weights`)
        this.model.setWeights(aggregatedWeights)
        const date = new Date()
        const modelName = `mnist-aggregated-model-${this.id}-${date.getTime()}`
        // await this.model.save(`localstorage://${modelName}`)
        // console.log(`Saved aggregated mode as ${modelName} in localStorage`)
        //sending aggregated weights to participants
        // const recepients = []
        
        // this.weights_queue.forEach((el,index)=>{
        //     recepients.push(el.peer_id)
        // })
         
        const finalWeights = this.model.getWeights()
        const arrayfied_weights =[]
        finalWeights.forEach((el,index)=>{
            arrayfied_weights.push(el.arraySync())
        }) 
        showMessage(`testing ${modelName}`)
        // testMnist(null,this)
        this.rounds-=1
        if(this.rounds>0){
            console.log(`rounds left: ${this.rounds}`)
            this.weights_queue = []
            const peer_ids = Object.keys(this.addressBook)
            this.selected_clients = _.sampleSize(peer_ids,Math.floor(Math.max(...[this.sampling_rate*this.K,1])))
            console.log(`Selected clients:${this.selected_clients.toString()}`)
            this.selected_clients.forEach(async (el,index)=>{
                await this.sendWeightsToPeer(el,this.epochs_per_client,this.id,this.noiseScale,this.rounds)
                sleep(5000)
            })
            testMnist(null,this)
        }else{
            console.log("completed all rounds")
            //final share aggregated weights
            Object.keys(this.addressBook).forEach((el,index)=>{
                this.sendEncryptedMessage(el,JSON.stringify({
                    "message_type":"sent-aggregated-weights",
                    "message_content":{
                        "weights":arrayfied_weights,
                        "sender":this.id
                    }
                }))
            })
            testMnist(null,this)
        }
        // testMnist(null,this)
    }
    async fetchInitializerDataset(url){
        const data = await tf.data.csv(
            url, {
                hasHeader: true,
                columnConfigs: {
                    label: {
                        isLabel: true
                    }
                }
             } 
        );
        return data
    }
    async initializeModel(){
        const client_training_data_url = `/mnist-federated-dataset/init.csv`
        const datasetLength = await getFederatedDatasetLength(client_training_data_url)
        console.log(`Dataset Length for local update: ${datasetLength}`)
        // showMessage(`Running local update on ${client_training_data_url}`)
        // showMessage(`Local update will run for ${epochs} epochs`)
        const trainData = await this.fetchInitializerDataset(client_training_data_url)
        trainData.take(1).forEachAsync((d) => {
            console.log("init DATA EXAMPLE:")
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
        }).shuffle(250).batch(64)//maybe remove the shuffle
        this.training_history = []
        await this.model.fitDataset((processedData),{
            epochs:1,
            callbacks:[
                this.fitCallbacks,
            ]
        })
    }
    async runLocalUpdate(epochs,initiator_id,scale,roundsleft){
        // const client_training_data_url = `/mnist-federated-dataset/client-${this.id}-train.csv`
        const client_training_data_url = `/mnist-federated-dataset/client-${this.id}-train.csv`
        // const client_training_data_url = `/mnist_non_iid/client-${this.peer.id}-train.csv`
        const datasetLength = await getFederatedDatasetLength(client_training_data_url)
        console.log(`Dataset Length for local update: ${datasetLength}`)
        showMessage(`Running local update on ${client_training_data_url}`)
        showMessage(`Local update will run for ${epochs} epochs`)
        const trainData = await tf.data.csv(
            // training_data_url, {
            client_training_data_url, {
                hasHeader: true,
                columnConfigs: {
                    label: {
                        isLabel: true
                    }
                }
             } 
        );
        // trainData.take(1).forEachAsync((d) => {
            // console.log("TRAINING DATA EXAMPLE:")
            // console.log(d)
        // })
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
        }).shuffle(250).batch(64)//maybe remove the shuffle
        this.training_history = []
        showMessage("Training start")
        showMessage((new Date().toTimeString()))
        await this.model.fitDataset((processedData),{
            epochs:epochs,
            callbacks:[
                this.fitCallbacks,
            ]
        }).then(async ()=>{
            showMessage("Training completed")
            showMessage((new Date().toTimeString()))
            const date = new Date()
            const modelName = `${this.id}-model-${date.getTime()}`
            // await this.model.save(`localstorage://${modelName}`)
            // showMessage(`Saved model as ${modelName} in localStorage`)
            showMessage(`sending updated weights to ${initiator_id}`)
            let updated_weights = this.model.getWeights()
            console.log(`ADDING NOISE TO WEIGHTS scale: ${scale}`)
            for(let i=0;i<updated_weights.length;i++){
                const weightsShape = updated_weights[i].shape
                console.log(`The weight's shape is:`)
                console.log(weightsShape)
                const noise = tf.randomNormal(weightsShape,0,scale) 
                updated_weights[i] = updated_weights[i].add(noise)
                noise.dispose()
            }
            const updated_weights_array = []
            updated_weights.forEach((el,index)=>{
                updated_weights_array.push(el.arraySync())
            })
            console.log(`===testing model with noise(scale:${scale}) after local update===`) 
            this.model.setWeights(updated_weights)
            console.log(`THIS PEER IS ${this.id}`)
            this.sendEncryptedMessage(initiator_id,JSON.stringify(
                {
                    "message_type":"sent-weights-for-aggregation",
                    "message_content":{
                        "weights":updated_weights_array,
                        "peer_id":this.id,
                        "datasetLength":datasetLength,
                        "roundsleft":roundsleft
                    }
                }
            ))
            // tf.tidy(async ()=>await testMnist(null,this))
            await testMnist(null,this)
        })
    }
    sendEncryptedTextMessage(id,content){
        this.sendEncryptedMessage(id,JSON.stringify({
            "message_type":"text-message",
            "message_content":content
        }))
    }
    encryptMessageForPeer(id,message){
        //add lock here
        if(!Object.keys(this.addressBook).includes(id)){
            console.log(this.addressBook)
            showError(`${id} not in address book.`)
            return false
        }else{
            const secret = this.addressBook[id].secret
            const key = cryptojs.createHash('sha256').update(String(secret)).digest('base64').substr(0, 32);
            console.log(`Key is: ${key}`)
            const algorithm = "aes256"
            const cipher = cryptojs.createCipher(algorithm,key)
            const encrypted = cipher.update(message,"utf8","hex")+cipher.final("hex")
            return encrypted
        }
    }
    decryptMessageFromPeer(id,encryptedData){
        if(!Object.keys(this.addressBook).includes(id)){
            showError(`${id} not in address book.`)
            return false
        }else{
            const secret = this.addressBook[id].secret
            const key = cryptojs.createHash('sha256').update(String(secret)).digest('base64').substr(0, 32);
            console.log(`Key is: ${key}`)
            const algorithm = "aes256"
            const decipher = cryptojs.createDecipher(algorithm,key)
            const decrypted = decipher.update(encryptedData,"hex","utf8")+decipher.final("utf8")
            return decrypted    
        }
    }
}
function connectToPeer(id){
    if(node){
        node.peer.connect(id)
    }
}
let node = null
const input_text = document.getElementById("id-input")
document.getElementById("init-peer-btn").addEventListener("click",async ()=>{
    const id = input_text.value
    node = new PeerNode(id);
    const addressBook = await node.db.getObject("addressBook")
    if(addressBook){
        node.addressBook = addressBook
    }
})
document.getElementById("connect-to-peer-btn").addEventListener("click",()=>{
    if(node){
        const peerId = prompt("enter Peer id")
        node.connectTo(peerId)
    }
})
document.getElementById("disconnect-from-peer").addEventListener("click",()=>{
    alert("not implemented")
})

document.getElementById("init-connect-to-peer").addEventListener("click",()=>{
    const peerId = prompt("enter peer id to init secure communication")
    if(peerId){
        const current_connections = Object.keys(node.peer.connections)
        const current_address_book = Object.keys(node.addressBook)
        if(current_connections.includes(peerId)&& !current_address_book.includes(peerId)){
            node.sendMessage(peerId,{
                message_type:"join-fed"
            })
        }
    }
})

document.getElementById("run-simulated-fed").addEventListener("click",simulatedFederatedMain)
document.getElementById("clear-local-storage").addEventListener("click",()=>localStorage.clear())
document.getElementById("connect-to-peers").addEventListener("click",()=>node.connectToPeers())
document.getElementById("run-federated-session-p2p").addEventListener("click",()=>{
    node.initiateFederatedSession(0.5,3,5,0.047,false)
})
document.getElementById("clear-indexed-db").addEventListener("click",async ()=>{
    const exisitng_databases = await indexedDB.databases()
    exisitng_databases.forEach((el,index)=>{
        indexedDB.deleteDatabase(index.name)
    })
    showMessage("Cleared local database")
})
async function getUniqueID(){
    const uniqueId = await biri()
    return uniqueId //unique ID limited to device.
}

const cryptojs = window.cryptojs
console.log("Ready to use crypto library")
console.log(cryptojs)

class LocalDatabase{
    constructor(name){
        this.name = name
        this.db = levelup(leveljs(name,{valueEncoding:"json"}))
    }
    async putObject(key,value){
        await this.db.put(key,JSON.stringify(value),err=>{
            if(err){
                showError(err.message)
            }else{
                showMessage(`Added key:${key} - value:${JSON.stringify(value)} to db:${this.name}`)
            }
        })
    }
    async getObject(key){
        let result = null
        try{
            result = JSON.parse(await this.db.get(key,{asBuffer:false}))
        }catch(e){
            result = null
        }
        return result 
    }
}