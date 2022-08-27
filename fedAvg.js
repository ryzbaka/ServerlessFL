// const tf = require("@tensorflow/tfjs")
console.log("starting fedavg script...")

class BP{
    constructor(input_dim, batch_size,lr, file_name){
        this.filename = file_name //WHAT IS FILENAME USED FOR???
        this.len = 0 //WHAT IS THIS USED FOR???
        this.lr = lr //learning rate
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
    
    backward_prop(label){
        //calculating derivate for w4
        const l_deri_out = self.output_layer - label
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

function loadDataset(){
    console.log("reading dataset")
    dfd.readCSV(
        "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
    ).then(data=>{
        dataset = data
        const print_head_btn = document.querySelector("#print_head_btn")
        print_head_btn.disabled = false
        console.log("done reading dataset")
    })
    
}

function main(){
   console.error("Main function not implemented yet.") 
}

const load_dataset_btn = document.querySelector("#load_dataset_btn")
load_dataset_btn.addEventListener("click",loadDataset)
const print_head_btn = document.querySelector("#print_head_btn")
print_head_btn.addEventListener("click",()=>dataset.head(5).print())
let dataset = null
