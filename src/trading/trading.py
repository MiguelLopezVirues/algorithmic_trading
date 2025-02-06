import pandas as pd
import requests
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Union, Tuple
from datetime import datetime, timedelta, timezone
from bson.objectid import ObjectId
from dateutil import parser

load_dotenv()
API_KEY = os.environ.get("ALPACA_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET")
ALPACA_AUTH = os.environ.get("ALPACA_AUTH")

db_user = os.environ.get("MONGO_DB_USER")
db_pass = os.environ.get("MONGO_DB_PASS")
db_host = os.environ.get("MONGO_HOST")
database = os.environ.get("MONGO_DB_NAME")
mongodb_options = os.environ.get("MONGO_OPTIONS")


headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": ALPACA_AUTH
}

clients_account_id = {"client_1": "dcfbefb1-cbaa-4e60-9cef-9db801b8eb13",
                      "client_2": "edafa6db-b501-4d69-95c8-b92793a3680c"}


from data_load import MongoDBHandler
db_handle = MongoDBHandler(db_user=db_user, db_pass=db_pass, host=db_host, options=mongodb_options)
db_handle.connect_to_database(database)
collection = db_handle.check_create_collection("trades")


# Accounts
def get_account_trading_details(account_id: str) -> Dict:
    """
    Retrieve trading details for a specific Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.

    Returns:
        Dict: A dictionary containing the account details.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/account"

    response = requests.get(url, headers=headers)

    print(response.json())

    return response.json()

def get_account_relationship_id(account_id: str) -> str:
    """
    Retrieve the ACH relationship ID for a specific Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.

    Returns:
        str: The ACH relationship ID associated with the account.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/accounts/{account_id}/ach_relationships"

    response = requests.get(url, headers=headers)

    relationship_id = response.json()[0]["id"]

    return relationship_id


def make_funding_transaction(account_id: str, relationship_id: str, amount: float) -> Dict:
    """
    Initiate a funding transaction for an Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.
        relationship_id (str): The ACH relationship ID for the account.
        amount (float): The amount to be transferred.

    Returns:
        Dict: A dictionary containing the transaction response details.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/accounts/{account_id}/transfers"

    payload = {
        "transfer_type": "ach",
        "direction": "INCOMING",
        "timing": "immediate",
        "relationship_id": f"{relationship_id}",
        "amount": str(amount)
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

    return response.json()


# Positions
def get_open_positions_client(account_id: str, symbol_id: Optional[str] = None) -> Dict:
    """
    Retrieve open positions for a specific Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.
        symbol_id (str, optional): The symbol identifier to filter positions.

    Returns:
        Dict: A dictionary containing the open positions information.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/positions"

    if symbol_id:
        url += f"/{symbol_id}"

    response = requests.get(url, headers=headers)

    print(response.json())

    return response.json()


def close_all_open_positions_client(account_id: str) -> Dict:
    """
    Close all open positions for a specific Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.

    Returns:
        Dict: A dictionary containing the response from the close request.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/positions"

    response = requests.delete(url, headers=headers)

    print(response.json())

    return response.json()





def close_symbol_position_client(account_id: str, 
                                 symbol: str, 
                                 magnitude: float, 
                                 mode: str = "percentage") -> Dict:
    """
    Close a specific symbol position for an Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.
        symbol (str): The symbol identifier of the position to close.
        magnitude (float): The quantity or percentage to close.
        mode (str, optional): The mode of closing ('percentage' or 'qty'). Defaults to "percentage".

    Returns:
        Dict: A dictionary containing the response from the close request.

    Raises:
        ValueError: If an invalid mode is specified.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/positions/{symbol}"

    if mode == "percentage":
        url += f"?percentage={magnitude}"
    elif mode == "qty":
        url += f"?qty={magnitude}"
    else:
        raise ValueError("The mode especified is incorrect. It should be either 'percentage' or 'qty'")

    response = requests.delete(url, headers=headers)

    print(response.json())

    return response.json()


# Orders
def get_order_by_id(account_id: str, order_id: str) -> Dict:
    """
    Retrieve a specific order by ID from an Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.
        order_id (str): The ID of the order to retrieve.

    Returns:
        Dict: A dictionary containing the order details.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/orders/{order_id}"

    response = requests.get(url, headers=headers)

    print(response.json())

    return response.json()


def get_list_of_orders(
    account_id: str,
    status: Optional[str] = "all",
    limit: Optional[int] = None,
    after: Optional[Union[str, datetime]] = None,
    until: Optional[Union[str, datetime]] = None,
    direction: Optional[str] = None,
    nested: Optional[bool] = None,
    symbols: Optional[str] = None,
    qty_below: Optional[Union[str, float]] = None,
    qty_above: Optional[Union[str, float]] = None,
) -> Dict:
    """
    Retrieve a list of orders for a specific account with optional filters.

    Args:
        account_id (str): The ID of the account for which to retrieve orders.
        status (str, optional): Order status to be queried. Can be 'open', 'closed', or 'all'. Defaults to 'open'.
        limit (int, optional): The maximum number of orders in the response. Defaults to 50, max is 500.
        after (Union[str, datetime], optional): A timestamp in ISO 8601 format or a datetime object. 
            The response will include only orders submitted after this timestamp (exclusive).
        until (Union[str, datetime], optional): A timestamp in ISO 8601 format or a datetime object. 
            The response will include only orders submitted until this timestamp (exclusive).
        direction (str, optional): The chronological order of the response based on submission time. 
            Can be 'asc' or 'desc'. Defaults to 'desc'.
        nested (bool, optional): If True, the result will roll up multi-leg orders under the 'legs' field of the primary order.
        symbols (str, optional): A comma-separated list of symbols to filter by (e.g., 'AAPL,TSLA').
        qty_below (Union[str, float], optional): Filter orders with a quantity below this value.
        qty_above (Union[str, float], optional): Filter orders with a quantity above this value.

    Returns:
        dict: A JSON response containing the list of orders matching the specified filters.
    """
    
    # Function implementation here
    pass
    # Base URL for the Alpaca Broker API
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/orders?"

    # Add status if provided
    if status:
        url += f"&status={status}"

    # Add limit if provided
    if limit:
        url += f"&limit={limit}"

    # Add after timestamp if provided
    if after:
        url += f"&after={after}"

    # Add until timestamp if provided
    if until:
        url += f"&until={until}"

    # Add direction if provided
    if direction:
        url += f"&direction={direction}"

    # Add nested flag if provided
    if nested:
        url += f"&nested={nested}"

    # Add symbols if provided (comma-separated list)
    if symbols:
        url += f"&symbols={symbols}"

    # Add qty_below if provided
    if qty_below:
        url += f"&qty_below={qty_below}"

    # Add qty_above if provided
    if qty_above:
        url += f"&qty_above={qty_above}"


    response = requests.get(url, headers=headers)

    return response.json()


def cancel_all_open_orders(account_id: str) -> Union[Dict, List[Dict]]:
    """
    Cancel all open orders for a specific Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.

    Returns:
        Union[Dict, List[Dict]]: The response containing cancellation results.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/orders"

    response = requests.delete(url, headers=headers)

    if response.status_code == 207:
        succesful_cancellations = [order["id"] for order in response.json() if order["status"] == 200]
        unsuccesful_cancellations = [order["id"] for order in response.json() if order["status"] != 200]

        print(f"The following orders were cancelled: {', '.join(succesful_cancellations)}.")

        print(f"\nUnsuccesful cancellations: {', '.join(unsuccesful_cancellations)}.")

        return response.json()



def cancel_open_order(account_id: str, order_id: str) -> Dict:
    """
    Cancel a specific open order for an Alpaca account.

    Args:
        account_id (str): The ID of the Alpaca account.
        order_id (str): The ID of the order to cancel.

    Returns:
        Dict: The response from the cancellation request.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/orders/{order_id}"

    response = requests.delete(url, headers=headers)

    if response.status_code == 204:
        print(f"Order '{order_id}' cancelled succesfully.")
    else:
        print("Request error.")

    return response.json()

def create_order_for_client(
    account_id: str,
    symbol: str,
    qty: Optional[Union[str, float]] = None,
    notional: Optional[Union[str, float]] = None,
    side: str = "buy",
    type: str = "market",
    time_in_force: str = "day",
    limit_price: Optional[Union[str, float]] = None,
    stop_price: Optional[Union[str, float]] = None,
    trail_price: Optional[Union[str, float]] = None,
    trail_percent: Optional[Union[str, float]] = None,
    extended_hours: bool = False,
    client_order_id: Optional[str] = None,
    order_class: str = "simple",
    tags: Optional[List[str]] = None,
    take_profit: Optional[Dict] = None,
    stop_loss: Optional[Dict] = None,
    commission: Optional[float] = None,
    commission_type: str = "notional",
    source: Optional[str] = None,
    instructions: Optional[str] = None,
    subtag: Optional[str] = None,
    swap_fee_bps: Optional[Union[str, float]] = None,
    position_intent: Optional[str] = None,
) -> Dict:
    """
    Create an order for a client's account.

    Args:
        account_id (str): The ID of the account for which to create the order.
        symbol (str): Symbol or asset ID to identify the asset to trade.
        qty (str, optional): Number of shares to trade. Can be fractional for market and day orders.
        notional (str, optional): Dollar amount to trade. Cannot be used with qty.
        side (str, optional): Represents the side of the transaction. Defaults to "buy".
        type (str, optional): The order type. Defaults to "market". Equity assets have the following options: limit, stop, stop_limit, trailing_stop.
        time_in_force (str, optional): The time the order will last without being fulfilled until cancellation. Defaults to "day".
        limit_price (str, optional): Required if type is 'limit' or 'stop_limit'.
        stop_price (str, optional): Required if type is 'stop' or 'stop_limit'.
        trail_price (str, optional): Required if type is 'trailing_stop' and trail_percent is not provided.
        trail_percent (str, optional): Required if type is 'trailing_stop' and trail_price is not provided.
        extended_hours (bool, optional): If true, order is eligible to execute in pre/post-market. Defaults to False.
        client_order_id (str, optional): A unique identifier for the order. Automatically generated if not provided.
        order_class (str, optional): The order class. Defaults to "simple".
        tags (list, optional): List of order tags (max 4).
        take_profit (dict, optional): Take profit order details.
        stop_loss (dict, optional): Stop loss order details.
        commission (float, optional): The commission to collect from the user.
        commission_type (str, optional): How to interpret the commission value. Defaults to "notional".
        source (str, optional): Source of the order.
        instructions (str, optional): Additional instructions for the order.
        subtag (str, optional): Subtag for the order.
        swap_fee_bps (str, optional): Swap fee in basis points.
        position_intent (str, optional): Represents the desired position strategy.

    Returns:
        dict: A JSON response containing the details of the created order.
    """
    url = f"https://broker-api.sandbox.alpaca.markets/v1/trading/accounts/{account_id}/orders"

    payload = {
        "symbol": symbol,
        "qty": qty,
        "notional": notional,
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "trail_price": trail_price,
        "trail_percent": trail_percent,
        "extended_hours": extended_hours,
        "client_order_id": client_order_id,
        "order_class": order_class,
        "tags": tags,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "commission": commission,
        "commission_type": commission_type,
        "source": source,
        "instructions": instructions,
        "subtag": subtag,
        "swap_fee_bps": swap_fee_bps,
        "position_intent": position_intent,
    }

    # Remove None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Succesfully placed order {response.json()['id']} for {response.json()['position_intent']} of the symbol {response.json()['symbol']}. See details in response.")

    return response.json()




# Database with alpaca
def create_lagged_closure_date(days: int, hours: int, minutes: int) -> datetime:
    """
    Create a future closure date based on specified time offsets.

    Args:
        days (int): Number of days to add.
        hours (int): Number of hours to add.
        minutes (int): Number of minutes to add.

    Returns:
        datetime: The calculated future datetime in UTC timezone.
    """
    closure_date=(datetime.today() + timedelta(days=days,hours=hours, minutes=minutes)).replace(tzinfo=timezone.utc)
    
    return closure_date


def create_and_register_order(
                            account_id: str,
                            symbol: str,
                            notional_amount: Union[str, float],
                            intent_open_price: float,
                            position_intent: str,
                            closure_date: datetime
                        ) -> Tuple[List[ObjectId], Dict]:
    """
    Create an order and register it in the database.

    Args:
        account_id (str): The ID of the Alpaca account.
        symbol (str): The symbol to trade.
        notional_amount (Union[str, float]): The notional amount for the order.
        intent_open_price (float): The intended open price for the position.
        position_intent (str): The intent description for the position.
        closure_date (datetime): The planned closure datetime for the position.

    Returns:
        Tuple[List[ObjectId], Dict]: A tuple containing the MongoDB insertion IDs and the order record.
    """
    
    order_response = create_order_for_client(account_id=account_id, symbol=symbol, notional=notional_amount, position_intent=position_intent) #, **argumentos_adicionales) # Could need additional parameters

    if order_response.get("id",None) == None:
        print(f"Order could not be placed due to {order_response.get('message', '_missing message_')}:")
        print(order_response)

    order_mongodb_record = [{"content": {
        "alpaca_order_id_open": order_response["id"],
        "alpaca_order_id_close": None,
        "alpaca_account_id": account_id,
        "asset_id": order_response["asset_id"],
        "order_creation_date": parser.isoparse(order_response["created_at"]),
        "planned_close_date": closure_date,
        "symbol": symbol,
        "position_intent": position_intent,
        "status_open": "PENDING",
        "status_close": None,
        "intent_open_price": intent_open_price,
        "notional_amount": notional_amount,
        "qty": None,
        "fulfilled_open_date": None,
        "fulfilled_open_price": None,
        "fulfilled_close_date": None,
        "fulfilled_close_price": None,
        "pct_profit_trade": None,
        "currency": "USD"
    }}]

    insertion_result = db_handle.insert_documents(collection_name="trades", documents=order_mongodb_record)

    if insertion_result.inserted_ids:
        print("Document inserted successfully with _id:", insertion_result.inserted_ids)
    else:
        print("Insertion failed.")
    
    return insertion_result.inserted_ids, order_mongodb_record[0]


def update_order_field(
        direction: str,
        alpaca_order_id: str,
        field_to_update: str,
        new_field_value: str
    ) -> None:
    """
    Update a specific field in a MongoDB order document.

    Args:
        direction (str): The order direction ('open' or 'close').
        alpaca_order_id (str): The Alpaca order ID to update.
        field_to_update (str): The field name to update.
        new_field_value (Any): The new value for the field.
    """
    # update status
    db_handle.db["trades"].update_one(
        {f'content.alpaca_order_id_{direction}': alpaca_order_id}, # Filter
        {"$set": {f'content.{field_to_update}': new_field_value}}  # Update
    )

def refresh_pending_status(direction: str, account_id: str, alpaca_order_id: str) -> None:
    """
    Refresh the status of a pending order in MongoDB based on Alpaca's current data.

    Args:
        direction (str): The order direction ('open' or 'close').
        account_id (str): The Alpaca account ID.
        alpaca_order_id (str): The Alpaca order ID to refresh.
    """
    # get order info from alpaca
    order_in_alpaca = get_order_by_id(account_id, alpaca_order_id)
    
    # update status field
    update_order_field(direction=direction,
                       alpaca_order_id=order_in_alpaca["id"], 
                       field_to_update=f"status_{direction}", 
                       new_field_value=order_in_alpaca["status"])

    # if filled, update price, date and quantity
    if order_in_alpaca["status"] == "filled":
        # update filled_price
        update_order_field(direction = direction, 
                            alpaca_order_id = order_in_alpaca["id"], 
                            field_to_update = f"fulfilled_{direction}_price", 
                            new_field_value = order_in_alpaca["filled_avg_price"])
        # update filled_date
        update_order_field(direction = direction, 
                            alpaca_order_id = order_in_alpaca["id"], 
                            field_to_update = f"fulfilled_{direction}_date", 
                            new_field_value = order_in_alpaca["filled_at"])
        
        if direction == "open":
            # update qty
            update_order_field(direction = direction, 
                                alpaca_order_id = order_in_alpaca["id"], 
                                field_to_update ="qty", 
                                new_field_value = order_in_alpaca["filled_qty"])
        else: # close
            # update pct_profit
            filled_open_price = float(db_handle.db["trades"].find_one({f'content.alpaca_order_id_close': alpaca_order_id})["content"]["fulfilled_open_price"])
            filled_closed_price = float(order_in_alpaca["filled_avg_price"])
            pct_profit_trade = filled_closed_price / filled_open_price - 1
            
            update_order_field(direction = direction, 
                                alpaca_order_id = order_in_alpaca["id"], 
                                field_to_update = "pct_profit_trade", 
                                new_field_value = pct_profit_trade)

def refresh_statuses(direction: str, account_id: str) -> None:
    """
    Refresh statuses of all pending orders in MongoDB for a specific direction.

    Args:
        direction (str): The order direction ('open' or 'close').
        account_id (str): The Alpaca account ID.
    """
    
    pending_orders_mongodb = list(collection.find({"$or": 
                                                   [{f'content.status_{direction}':'PENDING'},
                                                    {f'content.status_{direction}':'accepted'}]}))

    for order in pending_orders_mongodb:
        order = order["content"]
        refresh_pending_status(direction=direction, account_id=account_id, alpaca_order_id=order[f"alpaca_order_id_{direction}"])

def check_open_due_orders(closing_datetime: datetime) -> List[Dict]:
    """
    Retrieve open due orders from MongoDB that are scheduled to close after a specific datetime.

    Args:
        closing_datetime (datetime): The cutoff datetime for planned closure.

    Returns:
        List[Dict]: A list of dictionaries containing due order details.
    """
    due_order_list_mongodb = list(db_handle.db["trades"].find(
                {"$and":
                [{'content.planned_close_date': {"$gte": closing_datetime}}, # After closing time filter
                {'content.status_open': 'filled'}
                ]}))
    
    due_order_list_mongodb = [due_order["content"] for due_order in due_order_list_mongodb]
    
    return due_order_list_mongodb

def check_open_due_orders_today() -> List[Dict]:
    """
    Retrieve open due orders scheduled to close today.

    Returns:
        List[Dict]: A list of dictionaries containing today's due order details.
    """
    due_order_list_mongodb = check_open_due_orders(closing_datetime = datetime.today())

    return due_order_list_mongodb


def close_due_order_position(mongodb_order_dict: Dict) -> Dict:
    """
    Close a due order position based on MongoDB order data.

    Args:
        mongodb_order_dict (Dict): The MongoDB order document containing position details.

    Returns:
        Dict: The response from the position closure request.
    """
    # close qty amount in corresponding asset position
    response_close = close_symbol_position_client(account_id=mongodb_order_dict["alpaca_account_id"],
                                 symbol=mongodb_order_dict["symbol"], 
                                 magnitude=mongodb_order_dict["qty"], 
                                 mode = "qty")
    
    # initialize close order status
    update_order_field(direction="open", # IMPORTANT: direction="open" to use alpaca_order_id_open
                       alpaca_order_id = mongodb_order_dict["alpaca_order_id_open"],
                        field_to_update = "status_close", 
                        new_field_value = "PENDING")
    
    # update alpaca_order_id_close
    update_order_field(direction="open", # IMPORTANT: direction="open" to use alpaca_order_id_open
                       alpaca_order_id = mongodb_order_dict["alpaca_order_id_open"],
                        field_to_update = "alpaca_order_id_close", 
                        new_field_value = response_close["id"])
    
    return response_close

def cancel_open_order_mongodb(account_id: str, alpaca_order_id: str) -> None:
    """
    Cancel an open order in both Alpaca and update its status in MongoDB.

    Args:
        account_id (str): The Alpaca account ID.
        alpaca_order_id (str): The Alpaca order ID to cancel.
    """
    cancel_open_order(account_id = account_id,
                      order_id = alpaca_order_id)
    
    # initialize close order status
    update_order_field(direction="open", # IMPORTANT: direction="open" to use alpaca_order_id_open
                       alpaca_order_id = alpaca_order_id,
                        field_to_update = "status_open", 
                        new_field_value = "cancelled")