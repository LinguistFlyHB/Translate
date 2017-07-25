/**
 * Created by YZ on 2017/7/25.
 */

function trans(){
    var data=document.getElementById("rawLan").value;
    $ajax({
        type:"GET",
        success:function(){
            document.getElementById("newLan").innerText=data;
        }
    })

}
