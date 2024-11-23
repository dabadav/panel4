import unittest
from datetime import datetime
from collections import OrderedDict
from panel4.panel4 import Content, Visitor


class TestVisitor(unittest.TestCase):
    def setUp(self):
        # Create Content mock objects
        self.content1 = Content(id=1, text="Hey", vector=[])
        self.content2 = Content(id=2, text="Bye", vector=[])
        self.content3 = Content(id=3, text="Me", vector=[])

        # Initialize the Visitor object
        self.visitor = Visitor()

    def test_get_n_last_visited(self):
        # Add events to the visitor's history with timestamps
        self.visitor.update(content=self.content1, event="open", timestamp=1000, isrecommended=1)
        self.visitor.update(content=self.content1, event="close", timestamp=1001, isrecommended=1)
        self.visitor.update(content=self.content2, event="open", timestamp=1002, isrecommended=0)
        self.visitor.update(content=self.content2, event="close", timestamp=1003, isrecommended=0)
        self.visitor.update(content=self.content3, event="open", timestamp=1004, isrecommended=1)
        self.visitor.update(content=self.content3, event="close", timestamp=1005, isrecommended=1)
        self.visitor.update(content=self.content1, event="open", timestamp=1006, isrecommended=1)

        # Retrieve the last 2 visited contents
        result = self.visitor.get_n_last_visited(n=2)

        # Assert the contents are ordered by most recent timestamps
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, 1)  # Most recent (timestamp=1004)
        self.assertEqual(result[1].id, 3)  # Second most recent (timestamp=1003)

    def test_get_n_last_visited_all(self):
        # Add events to the visitor's history
        self.visitor.update(content=self.content1, event="open", timestamp=1001, isrecommended=1)
        self.visitor.update(content=self.content2, event="close", timestamp=1002, isrecommended=0)

        # Retrieve all visited contents
        result = self.visitor.get_n_last_visited(n=5)  # Request more than available

        # Assert the result contains all unique contents
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, 2)  # Most recent (timestamp=1002)
        self.assertEqual(result[1].id, 1)  # Second most recent (timestamp=1001)

    def test_get_n_last_visited_no_history(self):
        # Assert ValueError is raised when history is empty
        with self.assertRaises(ValueError):
            self.visitor.get_n_last_visited(n=2)
