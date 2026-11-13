import tempfile
import unittest
from pathlib import Path
from analytical_tools import arithmetic_answer,CSVDatabase

class ArithmeticTests(unittest.TestCase):
    def test_exact_decimal_arithmetic(self):
        self.assertEqual(arithmetic_answer('CALC: (0.1+0.2)*10',True),'3')
        self.assertEqual(arithmetic_answer('ceil(19/4)+sum([2,3])+2**3'),'18')
        self.assertEqual(arithmetic_answer('round(2/3, 2)'),'0.67')
        self.assertEqual(arithmetic_answer('```python\nCALC: 2+3\n```'),'5')
    def test_non_integer_and_unsafe_expressions(self):
        for expression in ['__import__("os").system("echo bad")','open("secret").read()','2**999999','[0]*10000000','True','1/0']:
            self.assertIsNone(arithmetic_answer(expression),expression)
        self.assertIsNone(arithmetic_answer('1/2',True))

class CSVTests(unittest.TestCase):
    def make(self,files):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        for name,text in files.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
        db=CSVDatabase(root);self.addCleanup(db.close);return db
    def test_quoted_columns_numeric_affinity_and_casefold(self):
        db=self.make({'Sales.csv':'product name,unit price,region\nWidget,4,ОМСК\nWidget,2,Москва\n'})
        self.assertEqual(db.execute('SELECT SUM("unit price") FROM sales WHERE region = \'омск\''),('4',None))
        self.assertEqual(db.execute('SELECT COUNT(*) FROM data WHERE "unit price" = \'4\''),('1',None))
    def test_join_and_cte(self):
        db=self.make({'nested/orders.csv':'id,quantity\n01,2\n02,3\n','catalog.csv':'id,price\n01,10\n02,20\n'})
        self.assertEqual(db.execute('WITH joined AS (SELECT quantity*price AS total FROM orders JOIN catalog USING(id)) SELECT SUM(total) FROM joined'),('80',None))
    def test_mixed_text_column_keeps_original_numeric_spelling(self):
        db=self.make({'x.csv':'value\n4.00\nN/A\n'})
        self.assertEqual(db.execute("SELECT COUNT(*) FROM data WHERE value='4.00'"),('1',None))
    def test_semicolon_in_literal_and_multistatement(self):
        db=self.make({'x.csv':'name,value\na;b,7\nx,3\n'})
        self.assertEqual(db.execute("SELECT value FROM data WHERE name='a;b'"),('7',None))
        self.assertIsNotNone(db.execute('SELECT 1; DROP TABLE data')[1])
        self.assertIsNotNone(db.execute('PRAGMA database_list')[1])
        self.assertIsNotNone(db.execute("SELECT load_extension('bad')")[1])
        self.assertEqual(db.execute('SELECT SUM(value) FROM data'),('10',None))
    def test_full_rows_not_preview(self):
        db=self.make({'x.csv':'v\n1\n2\n3\n100\n'})
        self.assertEqual(db.execute('SELECT SUM(v) FROM data'),('106',None))
        self.assertIsNotNone(db.execute('SELECT v FROM data')[1])
    def test_duplicate_columns_and_system_prefix(self):
        db=self.make({'sqlite_sales.csv':'a,a,\n1,2,3\n'})
        self.assertEqual(db.execute('SELECT a+a_2+column_3 FROM data'),('6',None))
    def test_limits_never_silently_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'x.csv').write_text('v\n1\n2\n3\n',encoding='utf-8')
            with self.assertRaises(ValueError):CSVDatabase(p,max_rows=2)
    def test_cyrillic_semicolon_csv(self):
        db=self.make({'продажи.csv':'Город;Сумма\nОмск;3\nОмск;4\n'})
        self.assertEqual(db.execute('SELECT SUM("Сумма") FROM "продажи" WHERE "Город"=\'омск\''),('7',None))

if __name__=='__main__':unittest.main()
