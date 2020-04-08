Type: `object`

<i id="#">path: #</i>

&#36;schema: [http://json-schema.org/schema#](http://json-schema.org/schema#)

**_Properties_**

 - <b id="#/properties/deaths">deaths</b> `required`
	 - Type: `number`
	 - <i id="#/properties/deaths">path: #/properties/deaths</i>
 - <b id="#/properties/source">source</b> `required`
	 - Type: `string`
	 - <i id="#/properties/source">path: #/properties/source</i>
 - <b id="#/properties/state">state</b> `required`
	 - Type: `string`
	 - <i id="#/properties/state">path: #/properties/state</i>
 - <b id="#/properties/date">date</b> `required`
	 - Type: `string`
	 - <i id="#/properties/date">path: #/properties/date</i>
 - <b id="#/properties/cases">cases</b> `required`
	 - Type: `number`
	 - <i id="#/properties/cases">path: #/properties/cases</i>
 - <b id="#/properties/counties">counties</b> `required`
	 - Type: `array`
	 - <i id="#/properties/counties">path: #/properties/counties</i>
		 - **_Items_**
		 - Type: `object`
		 - <i id="#/properties/counties/items">path: #/properties/counties/items</i>
		 - **_Properties_**
			 - <b id="#/properties/counties/items/properties/date">date</b> `required`
				 - Type: `string`
				 - <i id="#/properties/counties/items/properties/date">path: #/properties/counties/items/properties/date</i>
			 - <b id="#/properties/counties/items/properties/source">source</b> `required`
				 - Type: `string`
				 - <i id="#/properties/counties/items/properties/source">path: #/properties/counties/items/properties/source</i>
			 - <b id="#/properties/counties/items/properties/cases">cases</b> `required`
				 - Type: `number`
				 - <i id="#/properties/counties/items/properties/cases">path: #/properties/counties/items/properties/cases</i>
			 - <b id="#/properties/counties/items/properties/fips">fips</b> `required`
				 - Type: `string`
				 - <i id="#/properties/counties/items/properties/fips">path: #/properties/counties/items/properties/fips</i>
			 - <b id="#/properties/counties/items/properties/deaths">deaths</b> `required`
				 - Type: `number`
				 - <i id="#/properties/counties/items/properties/deaths">path: #/properties/counties/items/properties/deaths</i>

_Generated with [json-schema-md-doc](https://brianwendt.github.io/json-schema-md-doc/)_
