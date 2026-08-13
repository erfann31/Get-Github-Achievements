<!-- page 1 -->

[Figure: Cover page with illustration of a person working at a computer, Python logo, and text listing topics like Numpy, Pandas, KNN, Decision Tree, Naive Bayes, Machine Learning, and Deep Learning]

Handwritten pamphlet

Ali Nazarizadeh
Reyhane Mohammadi

Numpy
Pandas
KNN
Decision Tree
Naive Bayes
Machine Learning
Deep Learning

---

<!-- page 2 -->

[Figure: Handwritten Persian calligraphy on grid paper, including Bismillah, names, and a signature]

---

<!-- page 3 -->

[Figure: Illustration of a woman working on a laptop, drinking from a cup, with a speech bubble containing code brackets "{...}". Below, there is a cup of coffee icon with steam and the text "Numpy".]

---

<!-- page 4 -->

کتابخانه numpy: به ما این قابلیت رو میده که داده های چند بعدی رو داخل اون ذخیره کنیم و یکسری پردازش انجام بدیم.

* یک قالب یا یک بسته برای ما فراهم میکند، ساختارها یا ماتریس های چند بعدی که داده هایی که دارای پیچیدگی
بالایی در ذخیره کنیم.

این کتابخانه تعدادی هم تابع دارد.

ساختار یک بعدی و دو بعدی در numpy: ↓

ماتریس: یک داده ساختار دو بعدی است.
که شامل تعدادی سطر و ستون است ← سطر یک بُعد، ستون هم یک بُعد است.
۲۰ تا داده میتونیم ذخیره کنیم.
برای اینکه به داده ای یک خونه دسترسی داشته باشیم، باید
مختصات اون نقطه رو بنویسیم.
برای مثال: خونه (2, 1) یعنی سطری که اندیسش 2 هست
و ستونی که اندیسش 1 هست. :)

[Figure: A 4x4 grid representing a 2D array/matrix with axes labeled "عمودی ← سطر" and "افقی ← ستون". Coordinates (0,0), (2,1), and (3,4) are marked in cells, along with row indices (اندیس 0 to 3) and column indices (اندیس 0 to 4).]

آرایه: یک داده ساختار یک بعدی است.
فقط دارای یک سطر است.
شبیه به لیست (list) ها است.
اگه آرایه ها رو کنار هم قرار بدیم میشه یک داده ساختار دو بعدی (ماتریس). :)

سه بعدی: آیا داده ساختار سه بعدی هم داریم؟!
بله.
اگر همین شکل ماتریس رو به صورت سه بعدی بکشیم، میشه یک داده ساختار سه بعدی.
یعنی عمق هم بهش اضافه کنیم.

array4 = [[[6,18,12],[8,14,1],[12,22,4]]]
اگر جای این اعداد باز آرایه بود
میشه یک داده ساختار سه بعدی
پ!

به همین خوشمزگی :)

[Logo: Nahal]

---

<!-- page 5 -->

برای استفاده از numpy باید اونو import کنیم $\leftarrow$ import numpy as np

فراخوانی شد $\leftarrow$ برای راحتی اسمش به
اختصار تبدیل میشه

numpy: یک کتابخانه بسیار مهم هست.
وقتی آناکنودا رو روی سیستم نصب میکنیم، این کتابخانه هم نصب میشه و ما فقط باید فراخوانی کنیم، نیازی نیست مجدد نصب بشه.

برای کسب اطلاعات بیشتر میتونیم از help کمک بگیریم

ساخت داده ساختار یک بعدی (array):
برای اینکه یک داده ساختار آرایه تعریف کنیم باید از تابع array استفاده کنیم $\downarrow$

$array1 = np.array([1, 2, 3])$

$print(array1) \Rightarrow$ مقدار داده ساختار رو چاپ میکنه

$print(type(array1)) \Rightarrow$ نوع داده ساختار رو میگه

$print(array1.shape) \Rightarrow$ تعداد داره رو میگه $\leftarrow$ (3,) $\rightarrow$ وقتی بعد کاما چیزی نیست یعنی یک بعدی هست

$print(array1.ndim) \Rightarrow$ چند بعدی بودنش رو میگه

ساخت داده ساختار در بعدی:
برای اینکه یک داده ساختار دو بعدی تعریف کنیم بازهم باید از تابع array استفاده کنیم $\downarrow$
اما به جای یک عدد یک آرایه داخل داده ساختار قرار میدیم $\leftarrow$ چطوری؟

$array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])$

وقتی shape رو بگیریم $\rightarrow$ تعداد سطر و ستون رو میگه $\leftarrow$ (4, 3) $\rightarrow$ چهار تا سطر و سه تا ستون

فیلتر کردن numpy:
وقتی ستون ها رو فیلتر میکنم، یعنی همشو میخوام.
[0: 2] $\leftarrow$ یعنی سطر یک و دو، با همه ستون ها.

$array2[0: 2, 0: 1]$

از 2 تا یکی کمتر $\leftarrow$ از 0 تا یکی کمتر از 1

[[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
[10, 11, 12]]

خروجی چاپ میشه $\leftarrow$
[[1],
[4]]

---

<!-- page 6 -->

متن متد max و min:
توی numpy از این دو تا هم استفاده میشود.
ما میتونیم مثل اینارو هم بنویسیم $\leftarrow$ argmin و argmax
سطر رو پیمایش میکنه و به عدد که رسید اندیسش رو برمی گردونه.

> [!NOTE]
> np.min(array5)
> array4.min()

متد mean:
میانگین میگیره، همه رو جمع میکنه تقسیم بر تعداد میکنه.

متد std:
انحراف استاندارد رو بررسی میکنه.
بگینه هر چی این اعداد بزرگتر باشه، یعنی داده های ما پراکنده است.
نزدیک میانگین نیستند.

متد T:
تغییر سطر و ستون $\leftarrow$ سطر رو جای ستون و ستون رو جای سطر قرار میده.

متد arange:
بین 10 و 20 اعداد رو ایجاد میکنه با یک گام البته یکی کمتر از 20 $\leftarrow$ np.arange (10, 20)
میاد گام هارو دوتا دوتا میره جلو $\leftarrow$ np.arange(10,20,2)

متد uniform:
$g=np.random.uniform(1,10,[3,4])$
داخل یک ماتریس 3 در 4 $\leftarrow$ اعدادی بین 1 و 10

> مثلا من یکی داره ساختار تمرین
> میکنم $\leftarrow$ array5
> تابع بیشتر $\leftarrow$ np.
> تابع بهتر $\leftarrow$ array.

متد divide:
یک داره ساختار میگیره و یک عدد $\leftarrow$ np.divide (array5,2)
داده های 5 array رو تقسیم بر 2 میکنه.

متد abs:
قدر مطلق میگیره.

---

<!-- page 7 -->

وقتی متخرها رو داخل ماتریس می‌ریزیم با zeros هم راه مندریک نقطه هم هست، چرا؟
بخش اعشاری است که با نقطه می‌آید.
اگر بریزیم داخل یک متغیر و از تابع dtype استفاده کنیم می‌بینیم که نوعش اعشاریه $\leftarrow$ float

متد ones:
این متد هم مثل متد zeros عمل می‌کند؛ تنها تفاوتش که دارد این است که به جای 0، 1 چاپ میشه.

هنوزم با متدها باید سروکله بزنیم چه حوصله سربره $\ddot{v}$

for i in a:
    Print (i)
$\leftarrow$ اشاره میکنه به آرایه‌های داخل ماتریس

L1 = []
for i in a:
    for j in i:
        L1.append(j)
$\leftarrow$ اشاره میکنه به داده‌های داخل ماتریس

متد eye:
عددی که بهش میدیم یه ماتریس میسازه. $n \times n$
مثلاً $\leftarrow$ eye (3) ۳ در ۳ میشه که قطر 1 است.

متد full:
به این صورته $\leftarrow$ np.full([4, 5], 3)
همه داده‌ها ستون سطر

متد linspace:
سه تا مقدار می‌گیره $\leftarrow$ np.linspace(2, 8, 5)
میگه که بین عدد 2 و 8، 5 تا عدد استخراج کن.
فاصله بین اعداد مهمه $\leftarrow$ اگر 1.5 بود باید فاصله بین همه این باشه

متد random:
np.random.random([2, 4]) $\leftarrow$
یک ماتریس 2 در 4 است.
مقدارها: اعداد بین 0 و 1

متد sum: همه داده‌ها رو جمع میکنه.
حالا برای این sum می‌تونیم داده‌ها رو فیلتر کنیم و اون بخش رو که می‌خواهیم جمع کنیم.

array4 [1:3, 1:3]. sum()

متد ndenumerate:
مختصات رو داره روشون میده. و مقدار داره ها.

---

<!-- page 8 -->

ادامه کار کردن با numpy:

* اگر بخواهیم ما را یک خونه در دسترس داشته باشیم:
array2 [0] [1] $\Rightarrow$ خونه $(0,1)$

* ما می توانیم مقدار یک خونه رو تغییر بدیم:
array2 [0] [1] = 20

* یا اینکه مقدار یک سطر رو تغییر بدیم:
array2 [0] = [10, 20, 30]

$\downarrow$
تعداد داده ها دقیقاً با برابر داده ها عقبی باشد.

تغییر ماتریس (سطر و ستون):
ما می توانیم اون ماتریس رو به آرایه تبدیل کنیم با reshape.
اول می شماریم ببینیم چند تا داده داره: مثلا 12 تا.
بعد اون رو به یک آرایه ای که 12 تا داده داره تبدیل می شود.

یا اینکه می تونیم بگیم که اون رو به ماتریس با ۳ سطر و ۲ ستون تبدیل کن یا ۶ سطر و ۲ ستون.
مانند reshape $\ddot{\smile}$

**نکته**: باید توجه کنیم که حاصل ضرب اون دو تا عدد برابر بشه با تعداد داده های جدول به هر عددی نمیشه قرار داد.

متد Zeros:
وقتی یک عدد بهش میدیم به تعداد اون عدد برای ما صفر چاپ میکنه در قالب آرایه.
همین رو هم می تونیم در قالب ماتریس داشته باشیم.

$\checkmark$ ما وقتی می خواهیم یه آرایه یا ماتریس یک میلیونی درست کنیم و مقدارها شو تنظیم کنیم به طور پیش فرض
می آیم مقدار همه رو صفر قرار میدیم، و در طول کار تغییرش میدیم.
اگر بخوایم دستیه همه رو صفر بزاریم خیلی زمان بر است.

---

<!-- page 9 -->

[Figure: A cute cartoon character holding a container and reading a paper]

به همین خوشمزگی

متد all:
همه باید True باشه $\leftarrow$ True

متد any:
حداقل یک داده پیدا میشه که True باشه.

متد average:
میانگین میگیره.
میانگین ستونی $\quad np.average(array5, axis=0)$
میانگین سطری $\quad np.average(array5, axis=1)$

متد sum هم axis داره یعنی میتونیم جمع رو به صورت سطری و ستونی انجام بدیم.

متد median:
میاد میانه رو حساب میکنه.
یعنی یه عددی که نصف عدد کوچکتر، نصف عدد بزرگتر از اونه.

متد sort:
از کوچیکتر به بزرگتر مرتب میکنه.
این تابع هم axis میگیره که به صورت سطری و ستونی مرتب میکنه.

متد vstack:
دوتا داده ساختار میگیره که زیر هم قرار میده.
دومی رو زیر اولی قرار میده.

متد hstack:
داده ساختار رو کنار هم قرار میده.

متد‌های sqrt, cos, sin:
میاد روی تک تک داده‌ها اعمال میکنه.

من می‌تونم ۲ تا داده ساختار تعریف کنم و ببرم
توی یک متغیر و کلی‌ام عملیات ریاضی انجام بدم.
مثلاً: $-, +, *, /$ $\leftarrow$ اینارو خونه به خونه
انجام میده.
(یعنی مختصات یکی)

ورودی tuple $\uparrow$

---

<!-- page 10 -->

[Figure: Illustration of a woman working on a laptop while drinking from a cup, with a thought bubble containing JSON-like code {...}, and a separate drawing of a coffee cup with the text Pandas next to it]

Pandas

---

<!-- page 11 -->

[Figure: Handwritten notes titled "Pandas" covering basics of Pandas library, Series, DataFrame, and code examples]

# Pandas

کتابخانه Pandas: بیشتر کارهای آماری تحلیلی.
Pandas و numpy خیلی شبیه به هم هستند.
Pandas: آماری و تحلیل داده $\leftarrow$ برای حوزه علوم داده.
numpy: برای base ریاضیات استفاده می‌شود و قدرت خوبی دارد.

ما توی Pandas هم می‌تونیم داده‌های چند بعدی داشته باشیم.
توی Pandas به داده ساختارهای یک بُعدی می‌گیم Series $\leftarrow$ متردش هم به همین صورت هست.
این چیزی شبیه به آرایه در numpy است.

| index | |
| :---: | :---: |
| 0 | 12 |
| 1 | 8 |
| 2 | 6 |
| 3 | 15 |
| 4 | 18 |
| 5 | 10 |

1_D

[Figure: Pink arrow pointing to index column with text: "index قرار میگیرد - خودش به صورت خودکار"]

به داده ساختارهای دو بُعدی در Pandas می‌گیم DataFrame $\leftarrow$ با استفاده از این متد یک داده ساختار دو بُعدی می‌سازیم.
یک داده ساختار دو بُعدی دارای سطر و ستون.

| index | columns | | | |
| :---: | :---: | :---: | :---: | :---: |
| | a | b | c | d |
| 0 | 12 | 0 | 13 | 11 |
| 1 | 8 | 19 | 4 | 10 |
| 2 | 6 | 3 | 20 | 6 |
| 3 | 15 | 4 | 18 | 6 |
| 4 | 18 | 20 | 7 | 3 |
| 5 | 10 | 13 | 19 | 8 |

2_D

پانداس رو باید import کنیم $\leftarrow$ لازم نیست نصب کنیم.
پانداس یک کتابخانه پایه‌ای و مهم است.
و با آنوکوندا نصب می‌شود.

`import Pandas as pd`

`Series1 = pd.series([2,4,6,8])`

`Print(type(Series1)) $\leftarrow$ نوعش رو بگو`

`Print(Series1.shape) $\leftarrow$ تعداد داده‌ها`

`Print(Series1) $\leftarrow$ چاپ میکنه`

---

<!-- page 12 -->

Comma Separated Values $\leftarrow$ CSV فایل
یعنی داده ها با کاما از هم جدا شده.
این فایل هم مثل قبلیا با متد read خونده میشه.
و باید درس فایل رو بدیم و همراه با پسوند .CSV
آخرش هم به صورت خودکار سطر و ستون رو میده $\leftarrow$ 752x9

df.shape
سطر و ستون

df.columns
اندیس columns رو میده

df.iloc[0:10, 0:3]
اولی سطر، دومی ستون ← یکی کمتر *

df.iloc[2][3]
مقدار یک داده با دوتا انت $\downarrow$
df.loc[0][0:2]

متد values:
میاد داده ها رو نشون میده.

متد head:
اگر براش مقدار نفرستیم میاد 5 تای اول رو نشون میده.
اگر مقدار بدی به اون تکرار رو نشون میده.

متد tail:
اگر براش مقدار نفرستیم میاد 5 تا آخر رو نشون میده.
اگر مقدار بدی به اون تکرار از آخر نشون میده.

df["Age"] >= 70
False یا True

df.loc[df["Age"]>=70]
برای اینکه لیست افراد رو بیه، داخل براکت
قرار میدیم.
Python 2-78

متد loc:
خوبی که داره اینه که میتونیم مستقیم ستون ها رو چاپ کنه.
df.loc[4]["Age"]

متد sum:
df["Age"].sum() .جمع رو حساب میکنه

ما می تونیم از عملگر های گت برای منطقی
هم اینا استفاده کنیم $\leftarrow$ مثلاً:
بگیم کسایی که سنشون بزرگتر مساوی 40
و دپارت دارند.

متد min:
df["Age"].min() .کوچکترین رو پیدا میکنه

متد max:
df["Age"].max() .بزرگترین رو پیدا میکنه

---

<!-- page 13 -->

من میتونم داده ها رو تغییر بدم: $df["Age"][2]=25$

چون فقط یک ستونه تعداد سطرها رو ببینم: $df["Age"].shape$

یه بخشی از داده رو جمع کنم: $df.Age[10:15].sum$

میانگین سن افراد: $df["Age"].mean()$

میانه رو حساب میکنیم: $df["Age"].median()$

من میخوام که از اون جدول برای مثال ستون های Age و outcome چاپ بشه:

باید بزارم داخل دو تا براکت $df[["Age", "outcome"]]$

یا اینکه میتونیم متد mean رو روش اعمال کنیم: $df[["Age", "outcome"]].mean()$

حالا قسمت گریدا رو بالا می رسیم به میتونیم از all و any استفاده کنیم.

آیا همه ی سن ها بزرگ تر مساوی 20 است؟ $all(df["Age"]>=20) \quad True$

آیا حداقل یک نفر وجود دارد که 100 سالش باشد؟ $any(df["Age"]>=100) \quad False$

متد value_counts:

تعداد داده رو میشماره: $df["Age"].value_counts()$

متد describe:

میاد یکسری عملیات یا متد رو روی داده ها پیاده سازی میکنه.

مثلا من میخواهم فقط یکی از داده ها رو اعمال کنم:

$df.describe().loc['max']$

مثلا از این و فقط اندیس 1 رو می خواهم یک [1] بهش اضافه میکنم.

میانگین سن افراد: $df["Age"].mean()$

انحراف استاندارد ستون Age: $df["Age"].std()$

تعداد داده ها = count  
میانگین هر ستون = mean  
انحراف استاندارد = std  
کوچکترین = min  
بزرگترین = max  
یکسری تابع ریاضی = 25% 50% 75%

---

<!-- page 14 -->

`series2 = pd.series([2, 4, 6, 8], index=['a', 'b', 'c', 'd'])`

میتونیم اندیس ها رو تغییر بدیم

فیلتر کردن داده ها :
`series[1] -> اندیس 1`
`series[0:2] -> از 0 تا 1`
`series['d'] -> اندیس d`

چون اندیس حروفه میتونیم مستقیم بگیم value که اندیس d هست

یک راه دیگه هم برای دسترسی به اندیس ها وجود دارد:
میتونیم با نقطه tab به اندیس ها دسترسی داشته باشیم ← `series.a`

خوندن داده ها :
ما میتونیم داده ها رو بزاریم داخل یک داده ساختار دیگه و به data بگیم از اون گونه:

`L1 = [1, 3, 5, 7, 9]`

`series3 = pd.series(data = L1, index = ['a', 'b', 'c', 'd', 'e'])`

اندیس ها هم از این قابلیت میتونند استفاده کنند:

`L2 = [1, 3, 5, 7, 9]`

`series 4 = pd.series(data = L1, index = L2)`

من میتونم یک دیکشنری داشته باشم که data رو از اون گونه:

$$D1 = \left\{ \begin{matrix} 'a': 12, \\ 'b': 8, \\ 'c': 10, \\ 'd': 16, \\ 'e': 4 \end{matrix} \right\} \text{ value}$$

`index`

`series 5 = pd.series(data = D1)`
هم index هم value

نکته : وقتی میخواهیم یک دیکشنری رو وارد کنیم باید فقط data رو برابر دیکشنری قرار بدیم، اگه بیایم اندیس ها رو تغییر بدیم دیگه value ها رو نمشناسه ← میزنه NaN

نکته: اگر من برای data یک عدد ثابت بفرستم و به هر تعدادی که اندیس گذاشتم اون عدد برای همه اندیس ها تکرار میشه.

---

<!-- page 15 -->

من حتی میتونم داده هایی رو که داخل تایپ دارم بفرستم توی series:

```python
import numpy as np
array1 = np.array ([12,22,18,16,10])
Series 7 = pd.series (array1)
series 7
```
میخوام یک فیلتری رو اعمال کنم روی داده ها.
مثلا میخوام داده هایی رو که بزرگتر از ۱۲ هست رو فیلتر کنه.

```python
Series 7 > 12 -> True یا False
Series 7 [Series 7 > 12] -> اندیس و عدد رو نشون میده
```
متد mean هم قبلا بهش اشاره کردیم -> میانگین رو حساب میکنه.

```python
Series 7 [Series 7 >= series 7.mean ()] -> اعداد بزرگتر از میانگین رو میگه
```
DataFrame

برای ایجاد داده ساختار دو بعدی از متد DataFrame استفاده میکنیم.

```python
df1 = pd.DataFrame ([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
وقتی اجرا بشه یک جدول چاپ میکنه -> `df1`

| | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 1 | 2 | 3 |
| 1 | 4 | 5 | 6 |
| 2 | 7 | 8 | 9 |
| 3 | 10 | 11 | 12 |

```python
print(type(df1)) -> تایپ رو میگه
print(df1.shape) -> تعداد سطر و ستون رو میگه
$\downarrow \quad \downarrow$
سطر $\quad$ ستون
```
میتونیم columns رو تغییر بدیم، به این صورت:

```python
df2 = pd.DataFrame ([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], columns = ['a','b','c'])
`df2` چاپ میکنه $\leftarrow$ به جای 0 1 2 $\leftarrow$ abc
```
من میتونم یک داده ساختار دو بعدی از نوع array ایجاد کنم
و یک لیستی برای columns بسازم و بگم DataFrame از این دوتا داده ساختار بگیره.
نمونه سورش کن به $\leftarrow$ Python 2_18

---

<!-- page 16 -->

groupby: گروه بندی میکنه -> بر اساس (سن)
`df.groupby("Age").mean()`
مثلا میانگین همه کسایی که سنشون 21 تا 81 به ترتیب.
جدول نتیجه <- python 2_18

مثلا من میخوام فقط گروه کسایی که سنشون 21 هست رو حساب کنم:
`df[df["Age"]==21].mean()`

`df.groupby("Age").min()` <- کوچکترین مقدار هر ستون رو بر اساس سن
می تونیم min یا هر ویژگی دیگه هم اعمال کنم.

من دو تا ورودی میدم BMI و Age <- می خوام بر اساس BMI Age گروه بندی شده رو میانگین BMI رو بگیره.
`df[["Age", "BMI"]].groupby("Age").mean()`
اینجا هر تاچی می تونم بنویسم.

من به دو روش می تونم طول Age رو حساب کنم.
1. `df["Age"].value_counts().count()`
2. `len(df["Age"].value_counts())`

`df.iloc[0] = [10, 130, 75, 20, 0, 36, 0.36, 42, 1]`

می تونیم داده ها رو واسطه ایندکس رو رو تغییر بدیم یا هر سطر دیگه ای df
باید توجه کنیم که تعداد داده ها یکی باشه.

حالا اگر بخوام ستون رو تغییر بدم بمن تونم خودم دستی چند تا داده وارد کنم. پس از جمله for استفاده می کنم.

```python
import random

L1 = [random.randint(20, 90) for i in range (752)]

df["Age"] = L1

df
```
من اگر بخوام اسم ستون هارو تغییر بدم:
`df = df.rename(columns={'Age':'a'})`

df

---

<!-- page 17 -->

ذخیره سازی:
برای اینکه فایل CSV رو ذخیره کنم از متد `to_csv` استفاده میکنم.
و بعد آدرس به همراه پسوند رو میدم:
`df.to_csv("G:\\Dataset\\csv.csv")`

حالا می خوام فایل رو به صورت اکسل ذخیره کنم از متد `to_excel` استفاده میکنم.
`df.to_excel("G:\\Dataset\\Excel.xlsx")`

می خوام فایل اکسل رو بخونم از متد `read_excel` استفاده میکنیم.
`df_excel=pd.read_excel("G:\\Dataset\\Excel.xlsx")`

من میتونم به داده ساختار دو بُعُدی ایجاد کنم که نوع اون رو هم خودم انتخاب کنم.
مثلا: `dtype=float`
نمونه سورسی که به `python2-18`

$$\begin{matrix}
df.to\_ + tab \\
\downarrow \\
\text{با هم هرچند که بخوایم ذخیره میکنیم}
\end{matrix}$$

$$\begin{matrix}
pd.read + tab \\
\downarrow \\
\text{هر جور که می خوایم می خونیم}
\end{matrix}$$

مثلا دو تا داده ساختار دارم می خوام هر دو رو با هم ادغام کنم: `concat`
و توی دوتا براکت قرار میدیم.
`df4=pd.concat([df2,df3], axis=0)`
زیر هم قرار میده $\leftarrow$ 0 $\leftarrow$ کنار هم قرار میده $\leftarrow$ 1

من میتونم داده هایی یک ستون رو ببرم تو یک
داده ساختار دیگه مثل اینست. ♡
`df["ci"].apply(lambda x: x>=40`
`and x<=45)`
این x داره اشاره میکنه به تک تک داده هایی که
بین 40 و 45 داخل x.
و متد apply یک func میگیره.
این True و False نشون میده اگر بخوایم
مقدار روشون بده باید داخل براکت بزاریم
و قبلش اسم تابع رو میاریم.

می تونیم با حلقه for پیمایش کنم.
`for i in df["_____"]:`
`Print(i)`

باید بگم هر آنکه برابر 0 بود چاپ کن $\leftarrow$ رعایت ندارد.
باید بگم هر آنکه برابر 1 بود چاپ کن $\leftarrow$ رعایت دارد.
می تونیم len هم بگیریم.
`len` هم چاپ کنه.

می تونیم به جای 0 و 1 تغییر بدم؟ رعایت دارد و ندارد.
می خوام شمارم کسانی که رعایت دارد و ندارد:
`df["outcome"].value_counts()`

به همین خوشمزگی :)

---

<!-- page 18 -->

[Figure: Illustration of a woman working on a laptop while holding a cup, with a speech bubble containing code brackets {...}, a coffee cup icon, and the text "Machine Learning"]

---

<!-- page 19 -->

مفهوم یادگیری ماشین و علوم داده
ماشین لرنینگ
داده کاوی یا دیتا ماینینگ $\longleftarrow$ با استفاده از تحلیل داده‌ها به یک دانشی رسید
علوم داده یا دیتا ماینینگ

* مهم ترین دانش بشر چیست؟

[Figure: Diagram showing a 'نوزاد' (newborn) growing into a 'کودک' (child). The newborn points to 'بقیه' (others) and 'مادر' (mother) (فقط مادر رو می شناسه). The child points to 'بقیه', 'مادر', 'پدر' (father) (میلیم که بزرگتر میشه پدر (خواهر و برادر) هم می شناسه).]

[Figure: Diagram showing a 'پزشک' (doctor) evaluating patients. Left side: doctor points to 'بیمار' (patient) and 'سالم' (healthy). Right side: doctor points to 'بیمار', 'سالم کرونا کاسم کلودرد' [?], 'سالم' (healthy). Text on right: پزشک با پرسیدن سوال میتونه بگه سالم هستیم یا بیمار $\longleftarrow$ اگر بیماریم چه بیماری است. و تو طبقه دست‌های مختلف طبقه میکنه.]

طبقه بندی: انسان با طبقه بندی ستونه داده‌های یک مسئله رو مرتب کنه و بر اساس اون تعمیم بگیره.

چرا داده‌ها؟؟
$\downarrow$
اندازه‌گیری‌ها، جزئیات فضای مسئله
$\downarrow$
مسئله‌ء رخداد و مسئله‌ای که می‌خواهیم حلش کنیم.

چرا مدل‌سازی در قالب ریاضیات نه؟
$\downarrow$
چرا به جای مدل‌سازی ریاضی، به یادگیری ماشین و علوم داده رو آورده ایم؟
$\downarrow$
به خاطر تحمل مناسب به مسئله
$\downarrow$
مسائلی از پیچیدگی خاصی برخوردار است

مهم ترین دانش انسان؟
$\downarrow$
طبقه بندی
$\downarrow$
داده ها
$\downarrow$
دانش

---

<!-- page 20 -->

من برای اینکه مسائل رو حل کنیم باید عبارتیم برای غلبه بر جهل نسبت به مسئله:
تکرار کردن پرسید و ثبت و اندازه گیری $\leftarrow$ باید تکرار زیادی داده داشته باشیم و از خارج رو تکرار کنیم
تا برای مثال: بتونیم تشخیص بیماری دیابت رو تشخیص $\leftarrow$ مجموعه داده (DataSet) $\leftarrow$ تکرار زیادی از داده ها
بدیم.

[Figure: A table and a cup with a face]

* هر داره یک رکورد است:
که دارای یکسری ویژگی و خروجی است.
* هدف ما اینه که اون رویکردی که انسان داره برای حل مسائل به کامپیوتر هم بدیم.
$\leftarrow$ یعنی طبقه بندی $\leftarrow$ برای اینکه طبقه بندی کنیم نیازم داده داریم.
$\leftarrow$ که به اون مجموعه ای از داده ها میگویم دیتاست.

* این مسئله تشخیص چند رقمی بودن اعداد رو میخواهیم با یادگیری ماشین حل کنیم.
این رویکرد نسبت میدیم به یک بردار
میاد این اعداد رو نیکاسه
میره به همین نمودار؛ و ممکنه
مثلاً 1 قبل از 10 است $\leftarrow$ پس یک رقمه است.

[Figure: A cute mug with steaming hot liquid]

---

<!-- page 21 -->

مسئله جدید: باید به یک نمودار دو بعدی نگاشت بدهیم $\leftarrow$ فضای دو بعدی

| نتیجه | وزن | قد |
| :---: | :---: | :---: |
| معمولی | 68 | 166 |
| معمولی | 74 | 170 |
| چاق | 88 | 164 |
| چاق | 90 | 172 |
| معمولی | 75 | 178 |
| لاغر | 60 | 172 |
| لاغر | 65 | 180 |
| معمولی | 80 | 174 |
| چاق | 85 | 168 |
| لاغر | 72 | 181 |

[Figure: Scatter plot with two diagonal lines separating data points on a grid ranging from 60 to 98 on the x-axis and 164 to 182 on the y-axis]

* خروجی‌های مسئله ممکن مقدارهای زیادی داشته باشند.
$2$ تا خروجی $\leftarrow$ $2$ کلاسه $\leftarrow$ calssifictoin
$3$ تا خروجی $\leftarrow$ $3$ کلاسه $\leftarrow$ calssifictoin $\leftarrow$ یک نمونه (بالایی)
$4$ تا خروجی $\leftarrow$ $4$ کلاسه $\leftarrow$ calssifictoin
$\vdots$

[Figure: Three small scatter plots showing classification examples: "دو کلاسه" (two classes with a line), "دو کلاسه" (two classes with a circle), and "چهار کلاسه" (four classes with intersecting lines)]

---

<!-- page 22 -->

[Figure: Flowchart showing data, machine learning algorithm, and classification]

[Figure: Diagram mapping machine learning workflow steps]

---

<!-- page 23 -->

[Figure: handwritten flowchart detailing a machine learning workflow: splitting data into training (70%-80%) and testing (30%-20%) sets, feeding into a machine learning algorithm, training using data to set decision boundaries, producing a trained algorithm, making predictions, evaluating algorithm accuracy with branches for correct (درست) and incorrect (غلط) outcomes, leading to a formula metric True / (True + Negative)]

---

<!-- page 24 -->

[Figure: An illustration of a woman drinking from a cup while working on a laptop, with a speech bubble containing code brackets {...}, and a purple icon of a coffee cup with the text "KNN" next to it.]

---

<!-- page 25 -->

الگوریتم KNN
K نزدیکترین همسایه

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | خروجی |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | 0 |
| | | | | | | | | 1 |
| | | | | | | | | 1 |
| | | | | | | | | 0 |
| | | | | | | | | 1 |
| | | | | | | | | 1 |
| | | | | | | | | 1 |
| | | | | | | | | 1 |
| | | | | | | | | 0 |
| | | | | | | | | 1 |

* بهتره که ما داده هامون رو به چهار بخش تقسیم کنیم. $\leftarrow$
* برای مثال 70% train و 30% test که هر کدوم اینا باز دو بخش است. که یک ویژگی و یک خروجی دارد.
* همین اسم ها استاندارد هست و پایتون از ما می خواد که با همینا نامگذاری کنیم.

سوال: الگوریتم های یادگیری ماشین با کدوم بخش آموزش می بینند؟

پاسخ: x_train, y_train

سوال: با کدوم بخش ارزیابی میشه؟

پاسخ: x_test, y_test

* الگوریتم یادگیری ماشین با توجه به ویژگی ها (x_test) اول یک حدس می زند و بعد به y_test مراجعه می کند تا ببیند جواب درست است یا نه.

---

<!-- page 26 -->

```python
for i in range (30):
    Result = classifier.Predict (([i]))
    print(f"number{i}:{Result}")

* با یک حلقه for می‌ترسیم اعداد رو فیلتر کنیم و جواب رو داشته باشیم.
* خب ما اگر جواب رو داشته باشیم می‌بینیم که از یک تا نه رو به درستی حدس زده و از یازده تا سی
رو هم به درستی حدس زده.
اما ده رو جزو یک رتبه ها قرار داده.
خب پس چرا دقت رو گفت صد درصد؟!
چرا ده رو اشتباه حدس زده؟

* می‌خوایم اینو بگیم که همیشه دچار خطا میشه حتی وقتی دقت ۱۰۰ باشه.
اون دقت برای دنیایی است که آموزش دیده.
ولی توی مسائل دنیای واقعی دچار خطا میشه. ت

* ما در مسائل واقعی نمی‌آییم به طور دستی داده ها رو بهش بدیم.
در مسائل واقعی داده های زیادی داریم و نمیشه اینا رو به طور دستی وارد کنیم.
پس باید از یک فایلی مثل CSV بخونیم.

`df = pd.read_csv('E:\dataset\Dataset.CSV')`
فراخوانی میکنه، به صورت استاندارد دیتا رو می‌خونه.

$x = df.iloc [:,0:8].astype(float)$
برای اینکه مقدار اعشاری 
حفظ بشه.
ستون اول تا یکی کمتر 
از ۸
تمام سطرها

$x$

$y = df.iloc[:,8]$
ستون ۸
همه سطرها

$y$

* وقتی داده ها رو تفکیک کردیم در یک داده ساختار
ذخیره می کنیم.

---

<!-- page 27 -->

`import Pandas as pd` $\longrightarrow$ میرویم کتابخانه pandas برای ذخیره دادها است.

`x_train = pd.DataFrame ([[2], [8], [5], [9], [11], [18], [14], [13], [15], [26]])`
$\downarrow$
تولید یک داده ساختار دو بعدی ذخیره میشه

`y_train = pd.series ([1, 1, 1, 1, 2, 2, 2, 2])`
$\downarrow$
یک بعدی

`from sklearn.neighbors import KNeighborsClassifier`
$\downarrow$
از sklearn, neighbors رو فراخوانی میکنه. کتابخانه sklearn برای حل مسائل
کمک مون میکنه.

`classifier = KNeighborsClassifier (n_neighbors=3)` $\longrightarrow$ یک نمونه رو ساختم.
$\downarrow$
مقدار K

`classifier.fit (x_train, y_train)` $\longrightarrow$ متد fit نمونه رو آموزش داره.
$\downarrow$
با این متد اون نمونه اجازه داده رو
اعمال میکنه روی $x\_train, y\_train$

`x_test = pd.DataFrame ([[1], [3], [7], [17], [12]])`
`y_test = pd series ([1, 1, 1, 2, 2])`
$\longrightarrow$ داده های تست رو ساختم.

`Classifier.score (x_test, y_test)` $\longrightarrow$ با این متد ارزیابی میکنیم
و همینطور دقت اون کلاس رو میگیره.

`classifier.Predict ([[6]])` $\longrightarrow$ مقدار میگیره و جواب رو میده.

`classifier.Predict_Proba ([[20]])` $\longrightarrow$ به طور کلی بررسی میکند و یکی درصد کار رو بهش
نسبت می دهد.

`array ([[0., 1.]])` $\longrightarrow$ با اطمینان $100\%$ عضو کلاس B یا دور که

---

<!-- page 28 -->

مسئله تشخیص اعداد چند رقمی!

* خب ما دوتا مقدار داریم:
یکی K یکی number
اینجا چی هستند حالا؟
* K معیار به تکرار اون مقداری که میری
اعداد اطرافش رو پیدا میکنه.
* number که اینجا همش مقدار 3 رو دادیم ← می خواهیم بفهمیم بده که چند رقمی است.
با توجه به رأی الگوریتم 3 عدد یک رقمی است.

ما باید به این توجه کنیم که همیشه در یادگیری ماشین خطا داریم.

[Figure: Two number lines with labeled K values and number boxes]
K=3
Number = 3

K=5
Number = 3

---

<!-- page 29 -->

حالا باید $x\_train$ و $x\_test$ و $y\_train$ و $y\_test$ رو مشخص کنیم.

`from sklearn.model_selection import train_test_split`
این متد کمک میکنه تا داده ها رو مشخص کنیم.

`x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)`

داده تست چند درصد باشه؟

این یک مقدار دیگه هم میگیره `random_state=0` این رو برابر با عددی قرار میدیم، فرقی نداره چی عددی باشه. برای اینکه هربار خروجی جدیدی نده.

* جمع تعداد کل داده‌ها $\leftarrow$ $x\_test$ و $x\_train$ باید بشه.

کار پیش پردازش: ۷۰ یا ۸۰ درصد زمان ما برای پیش پردازش دیتا صرف میشه.

یعنی ما باید دیتا رو بررسی کنیم و اگر مقدار غیر عددی بود $\leftarrow$ عددی کنیم.

اگر ستونی لازم نبود $\leftarrow$ حذفش کنیم و...

ما با متد isnull این کار رو انجام میدیم.

`df.isna().sum()`

میاد اون ستونی که مقدار ندارد رو پیدا میکنه و تعداد جای خالی رو برای هر ستون میگه.

خب حالا من باید این مسئله رو هندل کنم با متد `fillna`.

`df.fillna(df.mean(), inplace=True)`

باید آبا باشه تا انجام بشه. میانگین میگیره و بخش‌های خالی را پر میکند.

---

<!-- page 30 -->

K عشره که فرد در نظر بگیریم.
چون اگر زوج باشه ممکنه با ملکیری چالش روبه رو بشه.
ما تقسیم حق با اکثریت است. اگر 4 در نظر بگیریم ممکنه دو تا عضو کلاس A و دو تا عضو کلاس B باشه.
پس دچار خطا میشه.

همونطور که گفتیم ملکیری ستون های هرز نیست و باید حذف بشه.
ما با متد drop این کار رو می کنیم:
`df = df.drop(['id'], axis=1)`

متد $\leftarrow$ `value_counts` میاد تعداد رو میگه ، یعنی چی؟
تعداد کسایی که برای مثال ممکنه تکرار کنن $\leftarrow$ `4861` `0` $\leftarrow$ خروجی درگیر
تعداد کسایی که برای مثال ممکنه تکرار کنن $\leftarrow$ `249` `1` $\leftarrow$ از خروجی ها چیند مورد وجود دارد.

حالا همینو میتونیم روی ستون های ویژگی (x) اعمال کنیم.

`x['gender'].value_counts()`

مقدار غیر عددی $\leftarrow$ `...` `Female`
`...` `male` $\leftarrow$ جای این همو دیگه دیگه ای میتونه باشه
`...` `Other`

من این مقدارهای غیر عددی رو می خوام با مقدار عددی آپدیت کنم.

`x['gender'] = x['gender'].replace('female', 1)`
`x['gender'] = x['gender'].replace('male', 2)`
`x['gender'] = x['gender'].replace('other', 3)`

---

<!-- page 31 -->

[Figure: Illustration of a woman sitting with a laptop, drinking from a cup, with a thought bubble containing code brackets {...}, a coffee cup icon, and the text "Decision Tree"]

Decision Tree

---

<!-- page 32 -->

الگوریتم Decision Tree (درخت تصمیم)

درخت تصمیم در خیلی از درسهای رشته کامپیوتر، در ریاضیات گسسته، ساختمان داده، طراحی الگوریتم،
هوش مصنوعی وجود داره .

درخت تصمیم : میاد ویژگی‌ها را به صورت شاخه شاخه میکنه، تا بتونه طبقه بندی کنه.
مقدار‌های ویژگی‌ها را به دسته های مختلفی طبقه بندی میکنه.

[Figure: A flowchart representing a decision tree structure with nodes and branches labeled with features and decisions (0, 1, 2)]

مسئله تشخیص تعداد رقم های اعداد (کدشت، درخت تصمیم)

[Figure: Number line from 1 to 18 with colored dots representing numbers]

[Figure: Flowchart with 'عدد' node branching into conditions like 'بقیه ترازها' and 'نیم‌ترازها' leading to results (1, 2)]

به همین خوشمزگی :)

---

<!-- page 33 -->

`import Pandas as pd` $\rightarrow$ کتابخانه پانداس رو فراخوانی میکنیم.
`from sklearn.model_selection import train_test_split`
$$\text{به تبدیل داده ها به train و test } \llcorner$$
`from sklearn.tree import DecisionTreeClassifier`
$$\text{ما قبلاً KNN رو اینو فراخوانی میکردیم. } \llcorner$$
الان از کتابخانه sklearn درخت تصمیم رو فراخوانی میکنیم.

* کار با درخت تصمیم مثل KNN است و کار باهاشون خیلی آسان است. فقط برای ساختن نمونه باید
درخت تصمیم فراخوانی بشه.
`classifier = DecisionTreeClassifier()`

مثل KNN با متد fit آموزش میشه.
با متد score دقتش سنجیده میشه.
با متد predict آزمایش میشه.
با متد predict_proba میگه که با چه اطمینانی حدس زده (درصد نسبت میده)
داده های test و train هم مثل الگوریتم KNN به درخت تصمیم داده میشه.

* با حلقه for میتونیم اعداد مثلاً ۱ تا ۳۰ رو آزمایش کنیم که ببینیم چند نمره است. $5$
`for in range(30):`
`    Result = classifier.Predict([[i]])`
`    Print(f"number{i}:{Result}")`

توی درخت تصمیم ما میتونیم دیتا رو از فایل هایی مثل اکسل بخونیم. $\llcorner$
`df = pd.read_excel('E:\dataset\my_dataset.xlsx')`

با متد value_counts تعداد مقدارهای هر ستون رو میگیم. $\llcorner$
و دقیقاً مثل الگوریتم قبلی K مقدارهای غیر عددی رو با مقدار عددی آپدیت میکنیم.

---

<!-- page 34 -->

باشد iloc هم مقدارهای خروجی و ویژگی ها رو از y و x میگیریم.
باشد train_test_split (y_train, y_test, x_train, x_test)= x رو وصل رو به اون چهار بخش x_train, x_test و y_train, y_test میگیریم
test_size رو مشخص میکنیم که چند درصد از داده برای test باشه.
random_state هم برابر یک عددی قرار میدیم که هر بار یک دقت جدید نده.

خب ما دیتاها رو توی سیستم خودمون داشتیم و فقط تکس آدرس میدیم تا اونو بخونه.
اما کتابخانه sklearn خودش دیتاسری دیتا داره، یعنی نیاز نیست دیتا رو دانلودش کنیم.

`from sklearn.datasets import load_breast_cancer`
(بیماری سرطان سینه)

`breast_cancer = load_breast_cancer()` -> یک نمونه ازش ساختیم

`breast_cancer.data` -> باعث میشه میتونیم به دیتاهاش دسترسی داشته باشیم

`breas_cancer.feature_names` -> اسم ویژگی ها رو میگه

`breas_cancer.target_names` -> اسم خروجی ها رو میگه

`df = pd.DataFrame(data=feature_names, columns=target_names)`
توی داخل دیتا فریم ذخیرش میکنیم.

`x = breas_cancer.data` -> کاری که میکنیم اینه که x و y رو
`y = breas_cancer.target` -> مشخص کنیم.

---

<!-- page 35 -->

[Figure: illustration of a woman working on a laptop while drinking from a cup, with a speech bubble containing code brackets and a cup of coffee icon next to the text "Naive Bayes"]

Naive Bayes

---

<!-- page 36 -->

الگوریتم naive bayes (طبقه بندیز)

طبقه بندیز: استفاده از احتمال جهت طبقه بندی داده های جدید.

[Figure: Diagram showing a new data point branching into probability of class A and probability of class B, bracketed together with "بیشترین احتمال"]

حالا اگر $10$ تا طبقه داشتیم ده بار احتمال رو حساب میکنیم.

فرمول ریاضی که برای طبقه بندیز وجود دارد:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

سوال پیش میاد: ما گفتیم که از ریاضیات استفاده نمیکنیم و به سمت علوم داده پیش میرییم.
این درسته و ما ریاضیات رو در مرحله دوم استفاده میکنیم.
من با استفاده از ریاضیات (فرمول ریاضی) ریاست رو تحلیل میکنم.
یعنی باید یه داده ای باشه تا من بیام از ریاضیات استفاده کنم.
داده $\Leftarrow$ الگوریتم های یادگیری ماشین $\Leftarrow$ تحلیل داده

* تمام مراحل کدنویسی طبقه بندیز هم دقیقا مثل در الگوریتم KNN و درخت تصمیم انجام میشود و تکراری است.
از متد `fit`, `Score`, `Predict`, `Predict_Proba` استفاده میشود.
داده های `test` و `train` رو هم یکجوری میسازیم.
کتابخانه `Pandas` و متد `train_test_split` رو فراخوانی میکنیم.

---

<!-- page 37 -->

اما برای ساخت یک نمونه، نیز رو فراخوانی می کنیم:

* `from sklearn.naive_bayes import GaussianNB`
* `Classifier = GaussianNB()`

* مثل در الگوریتم قبل می‌تونیم داده رو از فایل بخونیم با متد `read_csv` و با پسوند CSV.
* کارهای پیش پردازش هم دقیقا مثل دو الگوریتم قبلی انجام میشه.

نکته پایانی از این سه الگوریتم:

* ما برای یک دیتاست $\leftarrow$ با الگوریتم های مختلف $\leftarrow$ به جواب های مختلف میرسیم.
* الگوریتم های مختلف رو روی دیتا اعمال میکنیم $\leftarrow$ تا بالاترین دقت رو بدست بیاریم.
له و اون بالاترین دقت رو مبنا قرار میدیم.

* ما وقتی میخواهیم از متد Predict استفاده کنیم $\leftarrow$ به اون باید مقدار بدیم.
(یعنی برای هر ویژگی یک مقداری میدیم).
حالا ما میتونیم اونا رو داخل یک متغیر ذخیره کنیم و از اون اونا رو بخونیم.
یا اصلا با input از کاربر مقدار بگیریم.

به همین خوشمزگی

---

<!-- page 38 -->

[Figure: Illustration of a woman drinking from a cup while working on a laptop with a speech bubble containing code brackets {...}, alongside a cup of coffee icon and the text "Deep Learning"]

Deep Learning

---

<!-- page 39 -->

# شبکه عصبی و یادگیری عمیق

* اول از همه باید بگم که یکی نیستند؛ مباحث زیادی دارند و ریاضیاتی نسبتاً پیچیده‌ای دارند.
ما از مغز خودمون برای حل مسائل استفاده میکنیم.
توی مغز ما چیه؟ کثیری از شبکه‌های عصبی. از چی تشکیل شده؟ از میلیاردها نورون.
که با تک تک نورون دیگه در ارتباطه.
(نورون: کوچکترین واحد پردازشی در مغز)

دانشمندان اومدن فکر کردن و گفتن که اول با مدل سازی نورون میتونند اونا رو در کنار هم قرار
بدند و یک شبکه عصبی رو می سازند.

کار نورون: تعدادی سیگنال میگیره اون ها رو پردازش میکنه و خروجی رو میده به نورون های بعدی.
(به مثل تابع)

[Figure: A diagram of an artificial neuron showing inputs x1 and x2, weights w1 and w2, bias b, a summation symbol $\Sigma$, a transfer/activation function box, and output y = f(A).]

داخل یک متغیر (A) میریزیم $\rightarrow A = x1w1 + x2w2 + b$
باید از یک تابع فعال ساز رد کنیم.

* $x1$ و $x2$ ورودی های نورون هستند؛ همون داده های ما توی دنیاست.
* هرکدوم از اون ها یک ضریبی دارند که به اصطلاح بهش میگیم وزن ($w$).
* $x1$ ورودی اول، دارای وزن $w1$.
* $x2$ ورودی دوم، دارای وزن $w2$.

---

<!-- page 40 -->

* یک نورون متونه چندتا ورودی داشته باشه، به این صورت :
$A=x_{1}w_{1}+x_{2}w_{2}+...+x_{n}w_{n}+b$

تابع فعال ساز چیه؟
این تصمیم میگیره که نورون ها توی شبکه باشن یا نباشن.
یا اینکه خروجی به چه صورت باشه.
تابع هایی که وجود دارند تعدادشون کم نیست.
و این توابع که وجود دارند بین 0 و 1 یا 1- تا 1+ خروجی میدن.

مثال از توابع:
Sigmoid : اون ورودی هایی که میگیرن یا همون $x$ ها وقتی ضرب در وزن یا همون $w$ شن (پردازش هم انجام میشه)، خروجی بین 0 و 1 هست.
اکثرا در لایه آخر از این استفاده میکنیم.

[Figure: Sigmoid function graph and formula $f(x) = \frac{1}{1+e^{-x}}$]

Relu : اگه ورودی کوچکتر از صفر باشه، خروجی همون صفر است.
اگه ورودی بزرگتر از صفر باشه همون خورده ورودی رو برمیگردونه. $\max(0, x)$

ما توی لایه های مختلف شبکه های عمیق میایم از توابع Relu استفاده میکنیم.
میتونه تابع خوبی باشه.

Softmax : میاد با توجه به اون ورودی هایی که میگیره و تعداد کلاس هایی که توی دنیاست هست یکسری خروجی تولید میکنه.
مثلا اگه 5 کلاسه باشه میاد 5 تا خروجی تولید میکنه.
روی درصدی رو به هرکلاس اختصاص میده.
از این تابع برای لایه آخر استفاده میشه. اونی که درصد بیشتری دارد خروجی اصلی است.

[Figure: Softmax calculation pipeline with input vector, exponential formula, and output probability vector]

---

<!-- page 41 -->

خوب ما باید از کلاسی قانون پیروی کنیم:
باشد از Keras مدل Sequential یک نمونه بسازیم به برای مثال اسمش رو بزاریم model و لایه هام
باشد Dense می سازیم.
`model = Sequential()` ← نمونه ساختیم
`model.add (Dense(3, input_dim = 4, activation = 'relu'))`
تعداد نورون های لایه مخفی اول $\leftarrow$ تعداد ورودی ها $\leftarrow$ در متن ساختار شبکه عصبی و یادگیری عمیق
ما می توانیم به سمت یادگیری عمیق بریم. $\leftarrow$ relu ما از تابع
`model.add (Dense(2, activation = 'relu'))` $\leftarrow$ اینجا دیگه input_dim نداریم.
برای این هم میتونیم برای هر لایه از یک $\leftarrow$ چون ورودی ها فقط به لایه اول وصل می شوند.
تابع فعال ساز استفاده کرد.
محدودیتی در این مورد وجود ندارد.
`model.add (Dense(1, activation = 'sigmoid'))`
این یادتون نره که لایه آخر همیشه تابع فعال ساز sigmoid باشه.
`model.compile (loss = 'binary_crossentropy', optimizer = 'adam')`
این مقدار loss برای مدل های دو کلاسه است. $\leftarrow$ این مقدار هم میاردها و طاهای کمینه رو پیدا میکنه. (کمینه سازی)
`model.fit (x_train, y_train, epochs = 100)`
اینکه باهاش آشناسیم $\leftarrow$ ی مقدار دیگه ای هم که میگیره همین.
یعنیEpochs چندبار فرصت آموزش میدیم؟
ما اینجا اینو 100 قرار دادیم یعنی 100 بار فرصت آموزش میدیم.
`model.evaluate (x_test, y_test)` $\leftarrow$ ارزیابی (دقتش رو بررسی کنیم)

---

<!-- page 42 -->

نکته مهم: اگر شما شبکه عصبی دیدید که لایه‌های مخفیش بیشتر از 2 تا بود بهش شبکه عمیق است.
البته یعنی میتونیم به سمت یادگیری عمیق بریم.

شبکه های عصبی معمولی یا سطحی: نهایتاً دو لایه دارند.
شبکه های عصبی عمیق یا یادگیری عمیق: بیشتر از دو لایه دارند.
} یکی از تفاوت‌هاشون است.
  پس همه کار نیست.

`feed forward`: عملیاتی که از ابتدای ورودی تا انتهای خروجی انجام میشه.
`Back Propagation`: عملیاتی که از انتهای خروجی تا ابتدای ورودی انجام میشه.
چرا این عملیات برگشت رو انجام میده؟
چون بهتره آموزش ببینه، حالا یعنی چی آموزش ببینه؟
توی شبکه های عصبی آموزش دیدن به چیزیه حب تنظیم وزن‌ها (w) و بایاس‌ها (b)
نیست. البته وقتی عملیات اول انجام میشه یه خروجی میده، اون خروجی یه خطایی
دارو که این انجام میشه؛ که به اون خطا میگن $\leftarrow Loss function$

| $x1$ | $x2$ | $x3$ | $x4$ | $x5$ | $x7$ | $x8$ |
|---|---|---|---|---|---|---|
| | | | | | | | $\rightarrow Loss function=???$
| | | | | | | | $\rightarrow Loss function=???$
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | | $\rightarrow Loss function=???$

خب فهمیدیم شبکه عصبی چیه پس باید چجوری اونو پیاده‌سازی کنیم. حالا چجوری؟
مثل مدل‌هایی که تو جلسات قبل دیدیم به پیاده‌سازی شون خیلی ساده اس - مثلاً با کتابخانه sklearn.
حالا در شبکه های عصبی و یادگیری عمیق از کتابخانه Keras استفاده می‌کنیم چون اینقدر دسترسی هستن به کتابخانه‌های
دارن. البته Keras یکی از اوناس به تنسورفلو، پای تورچ و ...

---

<!-- page 43 -->

[Figure: Neural network diagram with feedforward and backpropagation arrows, mathematical formulas for nodes, and handwritten Persian notes]

feed forward $\longrightarrow$

[Figure: Neural network diagram showing nodes $x_1, x_2, x_3$ connected to nodes 1, 2, and then to node 3, with weights $w_1$ to $w_8$ and biases $b_1, b_2, b_3$]

$\longleftarrow$ Back Propagation

لایۀ اول یا ورودی

لایۀ دوم
لایۀ مخفی

خروجی
لایۀ آخر

(1) $f_1(x_1w_1 + x_2w_2 + x_3w_3 + b_1)$

(2) $f_2(x_1w_4 + x_2w_5 + x_3w_6 + b_2)$

(3) $f_3(w_7(1) + w_8(2) + b_3)$

* ما توی گیت‌های منطقی گفته بودیم
که همیشه خروجی یه گیت، ورودی یه گیت
دیگه باشه.

خطی‌ها این لایه رو حساب
نمی‌کنند، خیلی‌ها هم حساب
می‌کنند. ولی بهتره که
حساب نکنیم.

حالا این شبکه‌های عصبی می‌تونه خیلی مختلف باشه:
از تفکر لایه ، نورون و...

* نمونه‌هاش داخل دورۀ آقای تقوی‌زاده در فصل سوم $\Leftarrow$ python3_5 *

---

<!-- page 44 -->

آدرس میدیم تا دیتا رو بخونیم:

۲) ادامه پیش بینی مسئله قبلی مدل 1:

$df = pd.read\_csv('/content/drive/MyDrive/Datasets/python3\_5$
$/Heart\ Attack\ Data\ Set.csv')$

تا اینجا ثابته و باید باشه.
بعد از اونه که باید آدرس بدیم،
یعنی اسم پوشه و اسم دیتاست.

* و باز هم متدهایی که تکرارین و حتما یادمونه:
* head: برای دیدن 5 سطر اول دیتا
* shape: برای دیدن تعداد سطر و ستون دیتا
* iloc: برای مشخص کردن $x$ و $y$
* value\_counts: خروجی ها رو میبینیم که چند تا از چندتاست.
* train\_test\_split: داده های train و test رو هم دقیقا مثل الگوریتم هایی که قبل توضیح دادیم
انجام میشه (مقدار test\_size ، random\_state یادت نره بچ)

* اینکه چند تا لایه مخفی وجود داره و در هر لایه چند تا نرون وجود داره چیز ثابتی نیست.
کدهای اصلی اینجاست، خسته نشدی که؟ :))
البته یه کم جلو تر همینا بود اما یه توضیحات و پارامترهای جدید هست که دوباره آوردم.

$model = Sequential()$

$model.add(Dense(32, input\_dim = 13, kernel\_initializer="uniform", activation$
$= "relu"))$

این متد برای اضافه 
کردن لایه مخفی جه...

چیزی که جدیده اینه:
برای مقدار وزن ها و بایاس ها باید عددی قرار بدیم. این پارامتر هم برای همینه.
یعنی مقدار uniform اعداد اعشاری تصادفی ایجاد میکنه بین بازه صفر و یک عدم.
روش هایی جز این هم وجود داره که بین بازه های دیگه است.

$model.add(Dense(8, kernel\_initializer="uniform", activation = "relu"))$

$model.add(Dense(1, kernel\_initializer="uniform", activation = "sigmoid"))$

---

<!-- page 45 -->

* نکته مهم توی کار با Google Colab اینه که چطور و از کجا داده هامون رو بخونیم!
وقتی با Google colab کار میکنیم بهتره که داده هامون رو هم از یک فضای آنلاین بخونیم.
وقتی با جوپیتر کار می کردیم دیتارو روی سیستم داشتیم و بهش آدرس میدادیم و میخوندیم.
اما بهتره الان دیتارو توی فضایی مثل Google Drive ذخیره کنیم و از اونجا دیتا رو بخونیم.

حالا ما برای اینکه به گوگل درایو وصل بشیم کافیه این دو خط کد رو بنویسیم.

$$\left.\begin{array}{l}
\text{from google.colab import drive} \\
\text{drive.mount('/content/drive')}
\end{array}\right\} \text{همیشه ثابته}$$

* وقتی اجراش کنیم یه لینکی ظاهر میشه که کلیک میکنیم روش
و باید بهش اجازه دسترسی بدیم ← بعدش یه کدی کممون میاد ← باید کپی کنیم ← و جایی که ملکه پیست کنیم.
enter رو بزنیم ← وصل میشیم به گوگل درایو.

کتابخانه ها :

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Flatten

from sklearn.model_selection import StratifiedKFold

---

<!-- page 46 -->

حس مادیاستی که داریم در خود طبقه بندی می کنیم؛ یا طبقه بندی دوکلاسه یا طبقه بندی چند کلاسه.
* نکته: براساس تجربه و استانداردها:
طبقه بندی دوکلاسه
لایه آخر $\leftarrow$ `model.add (Dense (1, activation = 'sigmoid'))`
`model.compile (loss = 'binary_crossentropy', optimizer = 'adam')`
لایه آخر، ۱ نورون باشه. `sigmoid` تابع فعال ساز هم

اگه دیاتمون دوکلاسه بود این پارامترها با این مقدارها باشه. و در لایه آخر ۱ نورون باشه.
توابع فعال ساز هم sigmoid باشه.

طبقه بندی چند کلاسه
لایه آخر $\leftarrow$ `model.add (Dense (3, 4, ..., activation = 'softmax'))`
`model.compile (loss = 'categorical_crossentropy', optimizer = 'adam')`
توی طبقه بندی چند کلاسه هم بهتره این پارامترها با این مقدارها باشه. و در لایه آخر ۳, ۴ و... نورون باشه.
برای دوکلاسه همیشه از چند کلاسه هم رفت ولی برعکس نمیشه.

الگوریتم بهینه ساز $\leftarrow$ پیدا کردن بهترین وزن و بایاس:

Adam
SGD
RMSprop
Adamax
Adadelta
Adagrad
Nadam
Ftrl

ما برای کدزنی میریم سمت Google colab
چون فضای بهتری است از جوپیتر.
ما می خواهیم الگوریتم های یادگیری عمیق را پیاده سازی کنیم.
چون مدل های یادگیری عمیق با GPU کار می کنند و پردازش سنگینی دارند.
روی CPU هم میشه ولی سرعت کمی داره.
والبته به صورت دایم نمی تونیم ازش استفاده کنیم.

اول از همه توی منوی بالا میریم تو Runtime و گزینه Change runtime type را اون رو روی GPU تنظیم می کنیم.

---

<!-- page 47 -->

model.compile(loos="binary_crossentropy", optimizer="adam", metrics=
["accuracy"])

این پارامتر مربوط به دقت هست.

history = model.fit (x_train, y_train, epoch=100, batch_size=8)

الگو خروجی رو ببینید ۳.۵ python
کم میشه $\leftarrow$ خطا = loss
زیاد میشه $\leftarrow$ دقت = accuracy

دتا رو دسته دسته میکنید که
فضای کمتری بگیره به همه رو با هم وارد نمکنید.

سؤال پیش میاد:
مگه تو فاز Train یا آموزش هستیم، پس این دقت چیه؟
* ما این دقت رو ملاک قرار نمیدیم ولی بررسی میکنیم ببینیم تو هر تکرار وضعیت چه جوریه.

دقت رو میسنجیم و این دقت رو ملاک قرار میدیم. $\leftarrow$ model.evaluate(x_test, y_test)
یکسری اطلاعات نسبت به این معماری رو $\leftarrow$ model.summary()
میخونده.
توی خروجی یه پارامتر داره به اسم Param!
که برای هر لایه یه عددی قرار داره، اونا چین؟
تعداد ورودی‌های نورون ها و تعداد بایاس نورون ها.

پیش بینی جمله قبلی مدل 2:

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_Y = np_utils.to_categorical(encoded_Y)
dummy_Y = astype(int)

* توی مدل دوم توی لایه آخر دوتا نورون قرار میدیم. و باید این کد رو بنویسیم ویاسته
تمام کدهایش رمراحلش تکراریه. این دو کلاسه است
میتونه چند کلاسه هم باشد

$\begin{matrix}
10 & \leftarrow & \text{خروجی} & 01 \\
\downarrow & & & \downarrow \\
\text{عضو کلاس } 1 & & & \text{عضو کلاس } 2
\end{matrix}$

---

<!-- page 48 -->

* بعضی وقت‌ها شبکه عصبی خیلی درگیر جزئیات میشه و از اصل قضیه غافل میشه.
بخاطر همین ما میاییم تکنیک فراموشی بهش تزریق می کنیم با Dropout
یعنی اتصال ها و چند نورون رو قطع و خاموش میکنه.

`model.add(Dropout(0.2))`

* مثلا یه صف رو در نظر بگیرید.
برای مثال من یه چیزی در گوش کسی که انتهای صف هست میگم و اون به کسی که جلو واستانه،
تا برسه به گوش کسی که اول صف هست؛ به نظر تون اون حرف چقدر تغییر میکنه؟
لایه مخفی رمقدار ها هم همینطور هستند، یعنی از وقتی اولین نورون به آخرین برسه تغییری می کنند،
که این میتونه باعث تغییر دقت بشه.
پس برای اینکه تغییر زیادی نکند از BatchNormalization استفاده می کنیم.

`model.add(BatchNormalization())`

تمام ♡

mr. Ali Nazarizade
Reyhane Mohammadi
