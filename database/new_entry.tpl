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


<style type="text/css">

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    grid-gap: 1em;
    align-items: top;
    margin: 60px;
    padding: 5px;
}

input {
    grid-column: 2;
    margin: 0;
}

label {
    grid-column: 1/2;
    width: auto;
    margin: 0;
    text-align: right;
}

p {
    grid-column: 1/3;
    text-align: center;
    margin: 0px;
}

#save {
    width: 15%;
}

</style>