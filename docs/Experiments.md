Using model:

```dicta-il/dictabert-ner```

Running:

```uv run .\de_identify.py  --input .\dummy_data\test_deid_label_2.csv --model dicta-il/dictabert-ner --mode label```


Example of bugs in results can be found in the file *dummy_data\test_deid_label_2_deid_label.json*

This is what we get:
```  {
    "id": 12,
    "text": "אורי בן דוד ת.ז. 234567890 עבר ניתוח ב-25.11.2023 בבית החולים בני ציון.",
    "deid text": "[PER] ת.ז. [TIMEX]567890 עבר ניתוח ב-[TIMEX] בבית החולים [ORG] [FAC].",
    "entities": [
      {
        "start": 0,
        "end": 11,
        "text": "אורי בן דוד",
        "label": "PER"
      },
      {
        "start": 17,
        "end": 20,
        "text": "234",
        "label": "TIMEX"
      },
      {
        "start": 39,
        "end": 49,
        "text": "25.11.2023",
        "label": "TIMEX"
      },
      {
        "start": 62,
        "end": 65,
        "text": "בני",
        "label": "ORG"
      },
      {
        "start": 66,
        "end": 70,
        "text": "ציון",
        "label": "FAC"
      }
    ]
  },
```

Note that it mistakingly interprets the string ```234``` as a Time Expression. Thus the id number string ``` 234567890 ``` is converted to
``` [TIMEX]567890 ```

What is DUC ?
```Definition: Represents a specific Product or DUC (Product) in Hebrew language datasets.```
Possible bug:
```

    "text": "פנייה מאיימייל: cohen.david@gmail.com לקבלת מידע רפואי נוסף.",

      {
        "start": 6,
        "end": 10,
        "text": "מאיי",
        "label": "DUC"
      }
```

From the release notes:
```ג.נ. כאמור המודל כולו ניורוני בלי שום לקסיקון ולכן לעיתים רחוקות מודל הלמטיזציה חוזה לקסמה שאיננה מתאימה. אפשר לקרוא יותר כאן: ```

List of all token labels:
```all_labels = list(ner_pipe.model.config.id2label.values())```


```
['B-ANG', 'B-DUC', 'B-EVE', 'B-FAC', 'B-GPE', 'B-LOC', 'B-ORG', 'B-PER', 'B-WOA', 'B-INFORMAL', 'B-MISC', 'B-TIMEX', 'B-TTL', 'I-DUC', 'I-EVE', 'I-FAC', 'I-GPE', 'I-LOC', 'I-ORG', 'I-PER', 'I-WOA', 'I-ANG', 'I-INFORMAL', 'I-MISC', 'I-TIMEX', 'I-TTL', 'O']
```
