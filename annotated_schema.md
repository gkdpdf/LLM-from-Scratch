### **tbl_product_master**
```json
["This table contains detailed metadata for various products, including pricing, packaging, units, categories, and custom attributes. It is used for converting quantities between units, identifying promotions, and structuring product taxonomy. The table supports UI display logic and batch rules via flags and conversion factors, and is essential for enriching sales, stock, and shipment data with standardized product information.", 
[
  ["industy_segment_name : Name of the industry segment the product belongs to, datatype: string, <sample values: CAR, CAR>"],
  ["pack_size_name : Size of the product packaging, datatype: string, <sample values: MRP 20, MRP 10>"],
  ["base_pack_design_name : Descriptive name of the base packaging design of the product, datatype: string, <sample values: Takatak Namkeen Tomato 100 Gm, Fatafat Bhel MRP 10>"],
  ["industy_segment_id : Identifier for the industry segment, datatype: string, <sample values: Western Snacks_Out of Home, Namkeen_Out of Home>"],
  ["pack_size_id : Identifier for the pack size, combining size and industry segment, datatype: string, <sample values: MRP 20_Western Snacks_Out of Home, MRP 10_Namkeen_Out of Home>"],
  ["base_pack_design_id : Unique identifier for the base packaging design, datatype: string, <sample values: BIA7L670887A51556, BDM5A690290A10596>"],
  ["product_name : Full product description including name, size, and weight, datatype: string, <sample values: Takatak Namkeen Tomato 100 GM*7.2 KG NGP, Fatafat Bhel MRP 10|47 GM*11.75 KG>"],
  ["ptr : Price to retailer, indicating the cost price for retailers, datatype: float, <sample values: 15.2625, 7.4999>"],
  ["ptd : Price to distributor, indicating the cost price for distributors, datatype: float, <sample values: 14.39, 7>"],
  ["display_mrp : MRP shown on the display or packaging, datatype: integer, <sample values: 20, 10>"],
  ["mrp : Maximum Retail Price - final consumer price, datatype: integer, <sample values: 20, 10>"],
  ["alternate_product_category : Alternative category for the product, datatype: string, <sample values: Bridges, Mix/Chivra/Chanachur>"],
  ["product_id : ERP system identifier for the product, datatype: string, <sample values: FI088701000720000D, FD029000471175000D>"],
  ["is_promoted : Flag indicating if the product is currently promoted, datatype: boolean, <sample values: False, False>"],
  ["product_weight_in_gm : Weight of the product in grams, datatype: integer, <sample values: 0, 0>"]
]]
```


### **tbl_primary**
```json
["This table captures primary sales transactions between sellers(Superstockist) and buyers (Distributors),it tells the order quantity against every product & invoiced quantity that was billed against those orders.Invoiced quantities are the actual sales made by SuperStockist to Distributors.", 
[
["super_stockist_id : Unique identifier for the super stockist, an integer representing the seller entity. <sample values: 19000102>"],
  ["super_stockist_name : Name of the super stockist, a string indicating the seller's business name. <sample values: S B Markplus Private Limited-2>"],
  ["super_stockist_zone : Geographical zone of the super stockist, a string indicating the zone. <sample values: >"],
  ["super_stockist_region : Geographical region of the super stockist, a string indicating the region. <sample values: >"],
  ["super_stockist_state : State where the super stockist is located, a string indicating the state. <sample values: DELHI>"],
  ["distributor_id : Unique identifier for the distributor, an integer representing the buyer entity. <sample values: 460307>"],
  ["distributor_name : Name of the distributor, a string indicating the buyer's business name. <sample values: SAWARIYA TRADING 41496>"],
  ["distributor_zone : Geographical zone of the distributor, a string indicating the zone. <sample values: NORTH>"],
  ["distributor_region : Geographical region of the distributor, a string indicating the region. <sample values: DELHI>"],
  ["distributor_state : State where the distributor is located, a string indicating the state. <sample values: DELHI>"],
  ["channel_type : Type of sales channel, a string indicating the channel type. <sample values: GT>"],
  ["product_id : Unique identifier for the product, a string representing the product SKU. <sample values: FD016600220792001D>"],
  ["product_name : Name and description of the product, a string indicating the product details. <sample values: Bhavnagri Gathiya MRP 5|22 GM*7.92 KG>"],
  ["ordered_quantity : Quantity of the product ordered, an integer indicating the number of units. <sample values: 1800>"],
  ["short_close_qty : Quantity of the product that was not fulfilled, an integer indicating the shortfall. <sample values: 0>"],
  ["sales_order_date : Date when the sales order was placed, a date indicating the order date. <sample values: 04/10/25>"],
  ["bill_date : Date when the invoice was generated, a date indicating the billing date.datatype: date <sample values: 24/04/25>"],
  ["invoiced_total_quantity : Total quantity of the product sold, an integer indicating the invoiced units. <sample values: 1800>"]
]]
```


### **tbl_superstockist_master**
```json
["This table serves as a master reference for mapping Super Stockist names to their corresponding ERP IDs, facilitating the identification and referencing of super stockists across distribution and supply chain datasets. It is crucial for hierarchy-level reporting and regional inventory analysis.", 
[
    ["superstockist_name : The name of the super stockist, represented as a string. It is used to identify the super stockist in the supply chain. <sample values: Kansal Estate Private Limited, S B Markplus Private Limited-2, etc.>"],
    ["superstockist_id : The unique ERP ID associated with each super stockist, represented as an integer. This ID is used for referencing in various datasets. <sample values: 19000149, 19000102, etc.>"]
]]
```


### **tbl_distributor_master**
```json
["This table captures the mapping between distributors and their assigned super stockists in the Delhi region, including multi-level sales hierarchy data, distributor segmentation, channels, geotag, and ERP identifiers. It is useful for understanding the sales organization structure, tax jurisdictions, and distributor classification, enabling location-wise planning, supply chain alignment, and geo-segmented performance reporting.", 
[
    ["superstockist_name : Name of the super stockist assigned to the distributor, represented as a string. <sample values: S B Markplus Private Limited-2, ...>"],
    ["level6_position_user : Name of the user at level 6 in the sales hierarchy, represented as a string. <sample values: Vinayak Mathur, ...>"],
    ["level5_position_user : Name of the user at level 5 in the sales hierarchy, represented as a string. <sample values: Manoj Kumar Gaur, ...>"],
    ["level4_position_user : Name of the user at level 4 in the sales hierarchy, represented as a string. <sample values: Lokesh Baweja, ...>"],
    ["level3_position_user : Name of the user at level 3 in the sales hierarchy, represented as a string. <sample values: Ashwani Mudgil, ...>"],
    ["level2_position_user : Name of the user at level 2 in the sales hierarchy, represented as a string. <sample values: Mohammad Gaffar, ...>"],
    ["distributor_name : Name of the distributor, represented as a string. <sample values: SAWARIYA TRADING 41496, ...>"],
    ["distributor_erp_id : Unique ERP identifier for the distributor, represented as an integer. <sample values: 460307, ...>"],
    ["distributor_type : Type of distributor, represented as a string. <sample values: Sub Stockist, ...>"],
    ["state : State where the distributor operates, represented as a string. <sample values: DELHI, ...>"],
    ["distributor_segmentation : Segmentation category of the distributor, represented as a string. <sample values: GT, ...>"],
    ["distributor_channel : Channel through which the distributor operates, represented as a string. <sample values: GT, ...>"],
    ["city_of_warehouse_address : City where the distributor's warehouse is located, represented as a string. <sample values: WEST DELHI WEST DELHI, ...>"],
    ["temp_created_date : Temporary date of record creation, represented as a date in DD/MM/YY format. <sample values: 30/03/25, ...>"]
]
]



### **tbl_shipment**
```json
["This table records shipment transactions from supplying plants to customers (sold-to parties), detailing the logistical and billing aspects of product movement. It serves as a key source for tracking dispatched quantities, billing dates, and the relationship between plants, regions, and customers. The data can be used for supply chain performance monitoring, shipment reconciliation with sales records, and identifying customer-level shipment patterns.", 
[
  ["supplying_plant : Code of the plant responsible for supplying the goods, datatype: string, <sample values: HM51>"],
  ["sales_district : Geographical sales district associated with the shipment, datatype: string, <sample values: NORTH>"],
  ["sold_to_party : Unique identifier for the customer (buyer) receiving the shipment, datatype: integer, <sample values: 19000102>"],
  ["sold_to_party_name : Name of the customer (buyer) receiving the shipment, datatype: string, <sample values: S B Markplus Private Limited>"],
  ["city : City where the customer is located, datatype: string, <sample values: Delhi>"],
  ["material : Unique identifier for the product being shipped, datatype: string, <sample values: FD062900401000000D>"],
  ["material_description : Description of the product being shipped, datatype: string, <sample values: Palak Sev MRP 10|40 GM*10 KG>"],
  ["actual_billed_quantity : The quantity billed to the customer for the shipment, datatype: integer, <sample values: 8750, 1750, 2500>"],
  ["invoice_date : Date when the shipment was invoiced, datatype: date, <sample values: 07/05/25>"]
]
]

```
