
prevposx = 0
prevposy = 0


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

    var posx = joystick.deltaX()/50
    if(posx > 1){
        posx = 1
    }
    if(posx < -1){
        posx = -1
    }
    var posy = joystick.deltaY()/50
    if(posy > 1){
        posy = 1
    }
    if(posy < -1){
        posy = -1
    }


    if( posx != prevposx || posy != prevposy ){
        console.log("LOGLOG"+joystick+ " " + posx + " " + posy)
        prevposx = posx
        prevposy = posy
        $.post('/commands', {throttle: 0, angle : 0}, function() {
            console.log("dataDrive done")
        })
    }

}


$(document).ready(function(){


    resizeFunc()
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


$(window).on('resize', function(){
    resizeFunc()
 
})


function resizeFunc () {
    console.log($(window).width())


    wwidth = $(window).width()
    wheight = $(window).height()

    imar = $("#videofeed").width()/$("#videofeed").height()

    if(wwidth * imar < wheight){
        $("#imgcontainer").width($(window).width()).height('auto')
        $("#videofeed").width($(window).width()).height('auto')
        
    }else{
        $("#imgcontainer").height($(window).height()).width('auto')
        $("#videofeed").height($(window).height()).width('auto')
    }

   



}