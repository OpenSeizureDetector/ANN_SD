<head>
  <link rel="stylesheet" type="text/css" href="static/style.css">
  <nav class="nav_bar">
    <ul>
      <li><a href="/datasets">Datasets</a></li>
      <li><a href="/new">Add Entry</a></li>
      <li><a href="/delete">Delete Entry</a></li>
    </ul>
  </nav>
</head>


<body>
  <p>Add a new dataset to the database:</p>

  <form action="/new" method="GET" class="form-grid">

    <label for="name">Name: </label>
    <input type="text" size="100" maxlength="100" name="name" placeholder="Enter name of paper"><br>
    
    <label for="paper_url">Paper URL: </label>
    <input type="text" size="100" maxlength="100" name="paper_url" placeholder="Enter url of paper"><br>

    <label for="data_url">Data URL: </label>
    <input type="text" size="100" maxlength="100" name="data_url" placeholder="Enter url of data"><br>

    <label for="sensor_pos">Sensor Position: </label>
    <input type="text" size="100" maxlength="100" name="sensor_pos" placeholder="Enter sensor position"><br>

    <label for="sample_freq">Sampling Frequency: </label>
    <input type="text" size="100" maxlength="100" name="sample_freq" placeholder="Enter sampling frequency"><br><br>
    
    <input type="submit" name="save" value="save">
    
  </form>
</body>

