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

  <form action="/delete" method="GET">

    <label for="entry_id">ID of entry to be deleted: </label>
    <input type="number" size="20" maxlength="20" name="entry_id"><br>
    <input type="submit" name="submit" value="submit">
  </form>

</body>
