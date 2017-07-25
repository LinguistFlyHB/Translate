/**
 * Created by YZ on 2017/7/25.
 */

function trans(){
    var raw = document.getElementById("rawLan").value;
    $.ajax({
        type: "GET",
        data: raw,
        success: function(obj){
            var result = JSON.parse(obj)
            document.getElementById("newLan").innerText = result.result;
        }
    });

}
