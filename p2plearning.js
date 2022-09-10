class PeerNode{
    constructor(id){
        this.peer = new Peer(id)
        this.peer.on("open",(id)=>{
            showMessage(`This Peer ${id} connected to brokering server.`)
        })
        this.peer.on("disconnected",()=>{
            showError("Disconnected from brokering server")
        })
        this.locked = false
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
        this.peer.connections[id][0].send(JSON.stringify(data)) 
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
        if(this.locked){
            showError(`Did not respond to message: ${data} from ${connection.peer} (${this.peer.id} is locked.)`)
        }else{
            this.locked = true
            showMessage(`Locked ${this.peer.id}`)
            const messageObject = JSON.parse(data)
            if(messageObject.message_type==="join-fed"){
                //check address book for peer with id already exists
                if(Object.keys(this.addressBook).includes(connection.peer)){
                    showError(`${connection.peer} already in federation`)
                }else{
                    //if not exist, generate prime, generator and keys 
                    showMessage(`initiating DH key share`)
                    this.lock = true
                    this.addressBook[connection.peer] = {
                        "dhObject": cryptojs.createDiffieHellman(256), //delete after verify shared
                        "key-verified":false //set to true once shared key matched
                    }
                    showMessage("Generating key, prime and generator")
                    const keys = this.addressBook[connection.peer]["dhObject"].generateKeys()
                    showMessage("Generated key")
                    console.log("my key is:")
                    console.log(keys)
                    const prime = this.addressBook[connection.peer]["dhObject"].getPrime().toJSON().data
                    showMessage("Generated prime")
                    const generator = this.addressBook[connection.peer]["dhObject"].getGenerator().toJSON().data
                    showMessage("Generated generator")
                    showMessage(`Sending key, prime and generator to ${connection.peer}`)
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
                showMessage(`${this.peer.id} is generating its keys`)
                
                const myKeys = this.addressBook[connection.peer]["dhObject"].generateKeys()
                
                showMessage(`${this.peer.id} generated it's keys`)
                showMessage(`${this.peer.id} is computing secret`)
                
                const mySecret = this.addressBook[connection.peer]["dhObject"].computeSecret(otherKeys)
                
                // showMessage(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                // console.log(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                this.addressBook[connection.peer] = {
                    "secret":mySecret.toString()
                }
                
                this.sendMessage(connection.peer,{
                    "message_type":"join-fed-final-dh",
                    "keys":myKeys.toJSON().data
                })
            }else if(messageObject.message_type==="join-fed-final-dh"){
                // const otherKeys = this.arrayStringToArray(messageObject.keys)
                const otherKeys = messageObject.keys
                showMessage(`${this.peer.id} is computing its secret`)
                const mySecret = this.addressBook[connection.peer]["dhObject"].computeSecret(otherKeys)
                // showMessage(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                // console.log(`${this.peer.id}'s secret is ${mySecret.toString()}`)
                this.addressBook[connection.peer] = {
                    "secret":mySecret.toString()
                }
            }
            else if(messageObject.message_type==="ping"){
                this.sendMessage(connection.peer,{
                    "message-type":"ping",
                })
            }
            this.locked = false
            console.log(`${this.peer.id}'s lock has been unlocked`)
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
document.getElementById("init-peer-btn").addEventListener("click",()=>{
    const id = input_text.value
    node = new PeerNode(id);
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
async function getUniqueID(){
    const uniqueId = await biri()
    return uniqueId //unique ID limited to device.
}

const cryptojs = window.cryptojs
console.log("Ready to use crypto library")
console.log(cryptojs)