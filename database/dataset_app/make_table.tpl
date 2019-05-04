<head>
  <link rel="stylesheet" type="text/css" href="/static/style.css">  
  <nav class="nav_bar">
    <ul>
      <li><strong><a href="#">Datasets</a></strong></li>
      <li><a href="/new">Add Entry</a></li>
      <li><a href="/delete">Delete Entry</a></li>
    </ul>
  </nav>
</head>



<body>
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
</body>
