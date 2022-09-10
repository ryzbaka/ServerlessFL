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
        const splitString = s.split(",")
        const arr = []
        splitString.forEach(el=>{
            arr.push(parseInt(el))
        })
        return arr
    }
    handleData(connection,data){
        showMessage(`Received from ${connection.peer}: ${data}`)
        if(this.locked){
            showError(`Did not respond to message: ${data} from ${connection.peer} (${this.peer.id} is locked.)`)
        }else{
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
                    const prime = this.addressBook[connection.peer]["dhObject"].getPrime().toJSON().data
                    showMessage("Generated prime")
                    const generator = this.addressBook[connection.peer]["dhObject"].getGenerator().toJSON().data
                    showMessage("Generated generator")
                    showMessage(`Sending key, prime and generator to ${connection.peer}`)
                    this.sendMessage(connection.peer,{
                        "message-type":"join-fed-init-dh",
                        "keys":keys.toJSON().data,
                        "prime":prime,
                        "generator":generator
                    })
                }
            }else if(messageObject.message_type==="join-fed-init-dh"){
                //IMPLEMENTING
                const otherKeys = messageObject.keys
                const prime = this.arrayStringToArray(messageObject.prime)
                const generator = this.arrayStringToArray(messageObject.generator)
                console.log(`Other key ${otherKeys}`)
                console.log(`prime ${prime}`)
                console.log(`generator ${generator}`)
            }
            else if(messageObject.message_type==="ping"){
                this.sendMessage(connection.peer,{
                    "message-type":"ping",
                })
            }
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