import sys
import json

if __name__ == "__main__":

    lstPOSValues = []
    lstSimplePOSValues = []
    for strLine in sys.stdin:
        strPOSValue = strLine.strip()
        lstPOSValues.append({"tag": strPOSValue, "description": strPOSValue, "dispPos": strPOSValue})
        lstSimplePOSValues.append({"tag": strPOSValue, "description": strPOSValue})
 
    strPOSValues = json.dumps(lstPOSValues, indent=4)
    strSimplePOSValues = json.dumps(lstSimplePOSValues, indent=4)
    
    print("""
    export interface POSTag {
        tag: string;
        description: string;
        dispPos?: string;
    }

    export const POS: POSTag[] =
    """ + strPOSValues + \
    """

    export const SimplePOS: POSTag[] =
    """ + strSimplePOSValues)
