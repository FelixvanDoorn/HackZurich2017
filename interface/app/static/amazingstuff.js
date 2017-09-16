
var sendDriveData = function(manager)
{

    nipple = manager.get(0);
    console.log("ids"+manager.ids)
    if(nipple == undefined || nipple == false || nipple == null ){
        var xpos = 0;
        var ypos = 0;
    }else{
        var xpos = -nipple.frontPosition.x*2/nipple.options.size
        var ypos = -nipple.frontPosition.y*2/nipple.options.size
    }
    console.log("LOGLOG"+nipple+ " " + xpos + " " + ypos )
    $.post('/dataDrive', {throttle: ypos, angle : xpos}, function() {
        console.log("dataDrive")
    })

}

$(document).ready(function(){
   
    
    var options = {
        color : "Blue",          // distance to recycle previous joystick in
        fadeTime : 0
    };
   
    var manager = nipplejs.create(options);
    var timeoutID = window.setInterval(function(){
        
        sendDriveData(manager)
    }, 200);
    
})
