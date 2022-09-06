class PeerNode{
    constructor(id){
        this.peer = new Peer(id)
        this.peer.on("open",(id)=>{
            showMessage(`This Peer ${id} connected to brokering server.`)
        })
        this.addressBook = {}
        this.peer.on("connection",connection=>{
            //someone else initiated connection
            connection.on("open",()=>{
                showMessage(`*Connected to ${connection.peer}`)
            })
            connection.on("close",()=>{
                //won't trigger if the peer that initiated the peer disconnects by simply closing the tab
                showError(`*Disconnected from ${connection.peer}`)
            })
            //add new connection to addressBook
            this.addressBook[connection.peer] = {
                "connection":connection
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
            //potentially useful for updating number of participating peers
            showError(`Disconnected from ${connection.peer}`)
            //remove connection from address book
            // delete this.addressBook[connection.peer]
        })
        //add new connection to addressBook
        this.addressBook[connection.peer] = {
            "connection":connection
        }
    }
    getAddressBook(){
        console.log(`${this.peer.id}'s address book`)
        console.log(this.addressBook)
        return this.addressBook
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