import React from "react";
import {View, Modal, Text, ScrollView, FlatList} from "react-native";

export default class PricesScreen extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            gasData: this.fetchGasData(),
            loading: true
        }
    }

    async fetchGasData() {
        try  {
            let result = await fetch("http://fuelprice-server.azurewebsites.net/prices/trondelag");
            result = await result.json();
            this.setState(state => ({
                gasData: result,
                loading: false
            }))
        } catch(e) {

        }
    }

    render() {
        if (this.state.loading) return null;
        else {
            return (
                <View>
                    <Modal
                        animationType={"fade"}
                        transparent={true}
                        visible={!!this.props.navigation.getParam("imageTaken")}
                        onShow={() => {
                            setTimeout(() => {
                                this.setState(state => ({
                                    modalVisible: false
                                }));
                                this.props.navigation.setParams({imageTaken: false});
                            }, 2000)
                        }}
                    >
                        <Text style={{backgroundColor: "#90ee90", fontSize: 25}}>
                            Image sent successfully!
                        </Text>
                    </Modal>
                    <FlatList
                        data={this.state.gasData}
                        keyExtractor={ item => item.RowKey }
                        renderItem={ ({item}) =>
                            <View style={{borderWidth: 1, borderColor: "black"}}>
                                <Text>Time: {item.Timestamp}</Text>
                                <Text>Price: {item.price}</Text>
                                <Text>Fuel type: {item.fueltype}</Text>
                            </View>}
                    >
                    </FlatList>
                </View>
            )
        }
    }
}