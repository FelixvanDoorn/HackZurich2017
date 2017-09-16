
var sendDriveData = function(joystick)
{

    // nipple = manager.get(0);
    // console.log("ids"+manager.ids)
    // if(nipple == undefined || nipple == false || nipple == null ){
    //     var xpos = 0;
    //     var ypos = 0;
    // }else{
    //     var xpos = -nipple.frontPosition.x*2/nipple.options.size
    //     var ypos = -nipple.frontPosition.y*2/nipple.options.size
    // }

    var posx = joystick.deltaX()/5
    if(posx > 1){
        posx = 1
    }
    if(posx < -1){
        posx = -1
    }
    var posy = joystick.deltaY()/5
    if(posy > 1){
        posy = 1
    }
    if(posy < -1){
        posy = -1
    }

    console.log("LOGLOG"+joystick+ " " + posx + " " + posy)
    $.post('/dataDrive', {throttle: 0, angle : 0}, function() {
        console.log("dataDrive done")
    })

}

$(document).ready(function(){
   
    //var y = document.getElementsByClassName('container');
    
    // var options = {
    //   //  zone: y[0],
    //     color : "Blue",          // distance to recycle previous joystick in
    //     fadeTime : 10,
    //     mode: 'dynamic'
    // };
   
    // var manager = nipplejs.create(options);

    var joystick = new VirtualJoystick()
    var timeoutID = window.setInterval(function(){
        
        sendDriveData(joystick)
    }, 200);
    
})
