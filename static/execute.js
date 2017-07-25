/**
 * Created by YZ on 2017/7/25.
 */

function trans(){
    //var rawdata = document.getElementById("rawLan").value;
    $.ajax({

        type: "GET",
        //data: rawdata,
        //async:false,
        success: function(obj){
            var result = JSON.parse(obj)
            document.getElementById("newLan").innerText = result.result;
            alert(result.result)
        }
    });

}
