<p>The available datasets are as follows:</p>

<table border="1">
%for row in rows:
     <tr>
     %for col in row:
     	  <td>{{col}}</td>
     %end
     </tr>
%end
</table>
